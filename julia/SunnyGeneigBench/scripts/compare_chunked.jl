# chunked launch attack against the laptop performance gap.
#
# For each (batch, num_streams, chunk_size) strategy, runs 5-sample median
# timing on identical SW08 inputs and reports speedup vs PR#8. Single-launch
# baseline is included in each batch group as the reference.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CUDA
using Dates
using LinearAlgebra
using Statistics
using Printf
using SunnyGeneigBench
using Sunny

include(joinpath(@__DIR__, "setup_sw08.jl"))

const CUDAEXT = Base.get_extension(Sunny, :CUDAExt)
@assert CUDAEXT !== nothing && isdefined(CUDAEXT, :hegvd_batched!)
const pr8_hegvd! = CUDAEXT.hegvd_batched!

const SAMPLES = 5

# (batch, num_streams, chunk_size). num_streams=0 means "single-launch
# baseline" (don't go through run_kernel_chunked!).
const STRATEGIES = [
    (1024, 0, 0),
    (1024, 2, 512),
    (1024, 4, 256),
    (1024, 4, 64),
    (4096, 0, 0),
    (4096, 2, 2048),
    (4096, 4, 1024),
    (4096, 8, 512),
    (4096, 8, 64),
]

function median_min_max(v::AbstractVector{Float64})
    s = sort(v)
    return (median(s), s[1], s[end], std(v))
end

function time_pr8!(H_d, Σ_d)
    CUDA.synchronize()
    t0 = time_ns()
    pr8_hegvd!(H_d, Σ_d)
    CUDA.synchronize()
    return (time_ns() - t0) / 1e9
end

function time_single_launch!(solver, H_d, Σ_d, W, U, info)
    CUDA.synchronize()
    t0 = time_ns()
    SunnyGeneigBench.run_kernel!(solver, Σ_d, H_d, W, U, info)
    CUDA.synchronize()
    return (time_ns() - t0) / 1e9
end

function time_chunked!(solver, H_d, Σ_d, W, U, info, num_streams, chunk_size)
    CUDA.synchronize()
    t0 = time_ns()
    SunnyGeneigBench.run_kernel_chunked!(solver, Σ_d, H_d, W, U, info,
                                          num_streams, chunk_size)
    # run_kernel_chunked! already syncs each stream; one final ctx-level
    # synchronize closes the timing window.
    CUDA.synchronize()
    return (time_ns() - t0) / 1e9
end

println("chunked-launch benchmark — SW08 inputs, "
        * "PR#8 vs single-launch vs chunked")
println("Sunny: $(pkgversion(Sunny))  CUDA runtime: $(CUDA.runtime_version())  host: $(gethostname())")
println()

print("Building SW08 SpinWaveTheoryDevice... "); flush(stdout)
swt_d, swt_host, sys_enlarged, cryst = build_sw08_swt_device()
const N = 2 * Sunny.nbands(swt_d)
println("OK (N=$N)")

@printf("Constructing NVRTC solver (autotune mode, cache=Auto)... ")
flush(stdout)
solver = Solver(N; tuning=:autotune, cache=:auto)
@printf("BlockDim=%d, BPB=%d, smem=%d B\n",
        block_dim_x(solver), batches_per_block(solver), shared_mem_bytes(solver))
println()
flush(stdout)

# Build once per unique batch size; reuse across strategies that share batch.
batch_to_inputs = Dict{Int,Tuple}()
function get_inputs(batch::Int)
    haskey(batch_to_inputs, batch) && return batch_to_inputs[batch]
    H_template, Σ_template = build_sw08_matrices(swt_d, swt_host, sys_enlarged,
                                                  cryst, batch)
    W_buf    = CUDA.zeros(Float64,    N, batch)
    U_buf    = CUDA.zeros(ComplexF64, N, N, batch)
    info_buf = CUDA.zeros(Cint, batch)
    batch_to_inputs[batch] = (H_template, Σ_template, W_buf, U_buf, info_buf)
    return batch_to_inputs[batch]
end

# PR#8 timing per batch — we do this once per batch and reuse across strategies.
pr8_times_by_batch = Dict{Int,Float64}()
function pr8_median!(batch::Int)
    haskey(pr8_times_by_batch, batch) && return pr8_times_by_batch[batch]
    H_template, Σ_template, _, _, _ = get_inputs(batch)
    # warm-up
    let H = copy(H_template), Σ = copy(Σ_template); time_pr8!(H, Σ); end
    samples = Float64[]
    for _ in 1:SAMPLES
        H = copy(H_template); Σ = copy(Σ_template)
        push!(samples, time_pr8!(H, Σ))
    end
    med, _, _, _ = median_min_max(samples)
    pr8_times_by_batch[batch] = med
    @printf("  PR#8 baseline at batch=%d: median %.4f s (%d mat/sec) — %d samples %s\n",
            batch, med, round(Int, batch/med), SAMPLES,
            join([(@sprintf "%.4f" t) for t in samples], " "))
    flush(stdout)
    return med
end

# Eigenvalue cross-check vs PR#8: one sample per (batch, strategy). We
# compare ours's W output against PR#8's evalues_d on bit-identical inputs.
function eig_cross_check(solver, batch, num_streams, chunk_size)
    H_template, Σ_template, W_buf, U_buf, info_buf = get_inputs(batch)
    Hp = copy(H_template); Σp = copy(Σ_template)
    pr8_evals = Array(pr8_hegvd!(Hp, Σp))
    fill!(W_buf, 0.0); fill!(info_buf, 0)
    Ho = copy(H_template); Σo = copy(Σ_template)
    if num_streams == 0
        SunnyGeneigBench.run_kernel!(solver, Σo, Ho, W_buf, U_buf, info_buf)
    else
        SunnyGeneigBench.run_kernel_chunked!(solver, Σo, Ho,
                                              W_buf, U_buf, info_buf,
                                              num_streams, chunk_size)
    end
    CUDA.synchronize()
    ours = Array(W_buf)
    spot_idxs = unique(round.(Int, range(1, batch, length=min(8, batch))))
    max_rel = 0.0
    for k in spot_idxs
        a = sort(pr8_evals[:, k])
        b = sort(ours[:, k])
        for i in 1:N
            scale = max(abs(a[i]), 1.0)
            max_rel = max(max_rel, abs(a[i] - b[i]) / scale)
        end
    end
    return max_rel
end

# Run one strategy.
function bench_strategy(solver, batch::Int, num_streams::Int, chunk_size::Int)
    H_template, Σ_template, W_buf, U_buf, info_buf = get_inputs(batch)

    # Warm-up.
    let
        Ho = copy(H_template); Σo = copy(Σ_template)
        if num_streams == 0
            time_single_launch!(solver, Ho, Σo, W_buf, U_buf, info_buf)
        else
            time_chunked!(solver, Ho, Σo, W_buf, U_buf, info_buf,
                          num_streams, chunk_size)
        end
    end

    samples = Float64[]
    for _ in 1:SAMPLES
        fill!(W_buf, 0.0); fill!(info_buf, 0)
        Ho = copy(H_template); Σo = copy(Σ_template)
        if num_streams == 0
            push!(samples, time_single_launch!(solver, Ho, Σo,
                                                W_buf, U_buf, info_buf))
        else
            push!(samples, time_chunked!(solver, Ho, Σo,
                                          W_buf, U_buf, info_buf,
                                          num_streams, chunk_size))
        end
    end

    med, lo, hi, σ = median_min_max(samples)
    eig_rel = eig_cross_check(solver, batch, num_streams, chunk_size)

    return (
        batch = batch, num_streams = num_streams, chunk_size = chunk_size,
        med = med, lo = lo, hi = hi, σ = σ,
        mps = batch / med,
        samples = samples,
        max_eig_rel = eig_rel,
    )
end

# Run the grid.
results = NamedTuple[]
for (batch, ns, cs) in STRATEGIES
    label = (ns == 0) ? "single-launch" :
            "chunked, $(ns) streams, chunk=$(cs)"
    @printf("=== batch=%d  %s ===\n", batch, label)
    flush(stdout)
    pr8_median!(batch)            # establish baseline once per batch
    r = bench_strategy(solver, batch, ns, cs)
    @printf("  ours: med=%.4f s (range %.4f-%.4f, σ=%.4f) → %d mat/sec  eig_rel=%.2e\n",
            r.med, r.lo, r.hi, r.σ, round(Int, r.mps), r.max_eig_rel)
    @printf("  samples: %s\n", join([(@sprintf "%.5f" t) for t in r.samples], " "))
    flush(stdout)
    push!(results, r)
    println()
end

# Final table.
println("================ chunked-launch grid ================")
@printf("%-7s | %-37s | %-12s | %-9s | %-9s | %s\n",
        "batch", "strategy", "median (s)", "mat/sec", "vs PR#8", "eig_rel")
println("--------+---------------------------------------+--------------+-----------+-----------+----------")
for r in results
    label = (r.num_streams == 0) ? "single launch (baseline)" :
            @sprintf("chunked, %d streams, chunk=%d", r.num_streams, r.chunk_size)
    pr8_med = pr8_times_by_batch[r.batch]
    speedup = pr8_med / r.med
    @printf("%-7d | %-37s | %12.4f | %9d | %8.3fx | %.2e\n",
            r.batch, label, r.med, round(Int, r.mps), speedup, r.max_eig_rel)
end
println("=====================================================================")

# Save report.
const RESULTS_DIR = abspath(joinpath(@__DIR__, "..", "results"))
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)
const _today_str = Dates.format(Dates.now(), "yyyy-mm-dd")
const REPORT_PATH = joinpath(RESULTS_DIR,
    "chunked_launch_grid_$(gethostname())_$(_today_str).md")

open(REPORT_PATH, "w") do io
    println(io, "# iter1 chunked-launch grid — $(gethostname()), $(Dates.now())")
    println(io)
    println(io, "- Sunny: `$(pkgversion(Sunny))`")
    println(io, "- CUDA runtime: `$(CUDA.runtime_version())`")
    println(io, "- N = $(N) (SW08 √3×√3 kagome AFM, :dipole mode)")
    println(io, "- Solver: BlockDim=$(block_dim_x(solver)), BPB=$(batches_per_block(solver)), smem=$(shared_mem_bytes(solver)) B")
    println(io)
    println(io, "## PR#8 baselines")
    println(io)
    println(io, "| batch | PR#8 median | PR#8 mat/sec |")
    println(io, "|---:|---:|---:|")
    for batch in unique([r.batch for r in results])
        med = pr8_times_by_batch[batch]
        @printf(io, "| %d | %.4f s | %d |\n", batch, med, round(Int, batch/med))
    end
    println(io)
    println(io, "## Strategy grid")
    println(io)
    println(io, "| batch | strategy | median (s) | range | mat/sec | speedup vs PR#8 | max eig_rel |")
    println(io, "|---:|---|---:|---:|---:|---:|---:|")
    for r in results
        label = (r.num_streams == 0) ? "single launch (baseline)" :
                "chunked, $(r.num_streams) streams, chunk=$(r.chunk_size)"
        pr8_med = pr8_times_by_batch[r.batch]
        speedup = pr8_med / r.med
        @printf(io, "| %d | %s | %.4f | %.4f–%.4f | %d | %.3f× | %.2e |\n",
                r.batch, label, r.med, r.lo, r.hi,
                round(Int, r.mps), speedup, r.max_eig_rel)
    end
end
println()
println("results saved → $(REPORT_PATH)")
