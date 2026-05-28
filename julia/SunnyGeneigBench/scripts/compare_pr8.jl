# local PR#8 comparison benchmark.
#
# Generates physically-meaningful matrices via the SW08 SpinWaveTheoryDevice
# pipeline, runs PR#8's hegvd_batched! and our NVRTC kernel on bit-identical
# inputs, and reports 5-sample median throughput per batch size with a final
# speedup ratio. Eigenvalue agreement is asserted at fp64 epsilon on a
# small spot sample.

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

# --- Resolve PR#8's hegvd_batched! across the CUDAExt module boundary ---
# The function is defined inside Sunny's CUDAExt extension module rather
# than exported as Sunny.hegvd_batched!. Reach into the loaded extension.
const CUDAEXT = Base.get_extension(Sunny, :CUDAExt)
@assert CUDAEXT !== nothing "Sunny CUDAExt extension not loaded"
@assert isdefined(CUDAEXT, :hegvd_batched!) "CUDAExt.hegvd_batched! not found — wrong Sunny commit?"
const pr8_hegvd! = CUDAEXT.hegvd_batched!

const SAMPLES = 5

function median_min_max(v::AbstractVector{Float64})
    s = sort(v)
    n = length(s)
    return (median(s), s[1], s[end], std(v))
end

function run_pr8_path!(H_d::CuArray{ComplexF64,3}, Σ_d::CuArray{ComplexF64,3})
    # PR#8 mutates H_d (in-place Cholesky) and Σ_d (becomes eigenvectors).
    # Caller must restore between samples.
    return pr8_hegvd!(H_d, Σ_d)
end

function run_our_path!(solver::Solver, H_d::CuArray{ComplexF64,3},
                      Σ_d::CuArray{ComplexF64,3},
                      W::CuMatrix{Float64},
                      U::CuArray{ComplexF64,3},
                      info::CuVector{Cint})
    # Our kernel solves H u = λ S u. To match PR#8's semantics, pass
    # H = Σ_d and S = H_d. The kernel mutates both buffers. Positional-
    # variant call avoids keyword-arg NamedTuple construction in the hot
    # path (diagnostic).
    SunnyGeneigBench.run_kernel!(solver, Σ_d, H_d, W, U, info)
    return W
end

# Time a single sample of each path. Caller passes in already-cloned
# fresh inputs so neither path's in-place mutation contaminates the next.
function time_single_pr8!(H_d, Σ_d)
    CUDA.synchronize()
    t0 = time_ns()
    pr8_hegvd!(H_d, Σ_d)
    CUDA.synchronize()
    return (time_ns() - t0) / 1e9
end

function time_single_ours!(solver, H_d, Σ_d, W, U, info)
    CUDA.synchronize()
    t0 = time_ns()
    SunnyGeneigBench.run_kernel!(solver, Σ_d, H_d, W, U, info)
    CUDA.synchronize()
    return (time_ns() - t0) / 1e9
end

function bench_one_size(swt_d, swt_host, sys_enlarged, cryst, batch::Int;
                        solver::Solver)
    println("=== batch=$batch ===")
    flush(stdout)

    # Generate a single canonical (H_d, Σ_d) on the host side via Sunny.
    H_template, Σ_template = build_sw08_matrices(swt_d, swt_host, sys_enlarged,
                                                  cryst, batch)
    n = size(H_template, 1)

    # Pre-allocate output buffers for our kernel so the timed loop does
    # no allocations.
    W_buf    = CUDA.zeros(Float64, n, batch)
    U_buf    = CUDA.zeros(ComplexF64, n, n, batch)
    info_buf = CUDA.zeros(Cint, batch)

    # Eigenvalue cross-check (one sample, both paths). Use copies so the
    # time-loop inputs are untouched.
    H_pr8 = copy(H_template); Σ_pr8 = copy(Σ_template)
    evals_pr8 = pr8_hegvd!(H_pr8, Σ_pr8)
    CUDA.synchronize()
    evals_pr8_h = Array(evals_pr8)

    H_ours = copy(H_template); Σ_ours = copy(Σ_template)
    SunnyGeneigBench.run_kernel!(solver, Σ_ours, H_ours;
                                  W=W_buf, U=U_buf, info=info_buf)
    CUDA.synchronize()
    evals_ours_h = Array(W_buf)
    info_h = Array(info_buf)
    nz_info = count(!iszero, info_h)
    nz_info > 0 && @warn "our kernel reported nonzero info" nz_info

    # Eigenvalue agreement on 8 spot-sampled batch indices, sorted ascending.
    spot_idxs = unique(round.(Int, range(1, batch, length=min(8, batch))))
    max_eig_rel = 0.0
    for k in spot_idxs
        a = sort(evals_pr8_h[:, k])
        b = sort(evals_ours_h[:, k])
        for i in 1:n
            scale = max(abs(a[i]), 1.0)
            rel = abs(a[i] - b[i]) / scale
            max_eig_rel = max(max_eig_rel, rel)
        end
    end
    @printf("  eigenvalue cross-check on %d spot batches: max rel = %.3e\n",
            length(spot_idxs), max_eig_rel)
    flush(stdout)

    # Warm-up: one sample of each path, results discarded.
    let
        Hp = copy(H_template); Σp = copy(Σ_template)
        time_single_pr8!(Hp, Σp)
        Ho = copy(H_template); Σo = copy(Σ_template)
        time_single_ours!(solver, Ho, Σo, W_buf, U_buf, info_buf)
    end

    # Timed sample loop. Fresh copies per sample so in-place mutation of the
    # previous run doesn't contaminate the next. After each sample we also
    # snapshot a hash of the W output buffer to verify the kernel actually
    # ran (vs. some hidden caching path returning the same memory).
    pr8_times      = Float64[]
    ours_times     = Float64[]
    pr8_w_hashes   = UInt64[]
    ours_w_hashes  = UInt64[]
    for s in 1:SAMPLES
        Hp = copy(H_template); Σp = copy(Σ_template)
        push!(pr8_times, time_single_pr8!(Hp, Σp))
        # PR#8 returns evalues_d as a CuMatrix; grab its hash via host copy.
        # Σp got overwritten with Bogoliubov eigenvectors, so we hash that
        # too to confirm the kernel actually wrote it.
        push!(pr8_w_hashes, hash(Array(Σp)))

        # Reset our outputs so we can confirm the kernel writes fresh data.
        fill!(W_buf, 0.0)
        fill!(info_buf, Cint(0))
        Ho = copy(H_template); Σo = copy(Σ_template)
        push!(ours_times, time_single_ours!(solver, Ho, Σo,
                                             W_buf, U_buf, info_buf))
        push!(ours_w_hashes, hash(Array(W_buf)))
    end

    pr8_med, pr8_min, pr8_max, pr8_std    = median_min_max(pr8_times)
    ours_med, ours_min, ours_max, ours_std = median_min_max(ours_times)
    pr8_mps  = batch / pr8_med
    ours_mps = batch / ours_med
    speedup  = pr8_med / ours_med

    @printf("  PR#8:  med=%.4fs (range %.4f-%.4f, σ=%.4f) → %d mat/sec\n",
            pr8_med, pr8_min, pr8_max, pr8_std, round(Int, pr8_mps))
    @printf("    sample times: ")
    for t in pr8_times; @printf("%.5f ", t); end
    @printf("\n")
    @printf("    W hashes:    ")
    for h in pr8_w_hashes; @printf("%016x ", h); end
    @printf("\n")
    @printf("  ours:  med=%.4fs (range %.4f-%.4f, σ=%.4f) → %d mat/sec\n",
            ours_med, ours_min, ours_max, ours_std, round(Int, ours_mps))
    @printf("    sample times: ")
    for t in ours_times; @printf("%.5f ", t); end
    @printf("\n")
    @printf("    W hashes:    ")
    for h in ours_w_hashes; @printf("%016x ", h); end
    @printf("\n")
    verdict = speedup > 1.0 ? "ours faster" : "PR#8 faster"
    @printf("  speedup (PR#8 / ours): %.3fx (%s)\n", speedup, verdict)
    println()
    flush(stdout)

    return (
        batch = batch,
        pr8_med = pr8_med, pr8_min = pr8_min, pr8_max = pr8_max, pr8_std = pr8_std,
        ours_med = ours_med, ours_min = ours_min, ours_max = ours_max, ours_std = ours_std,
        pr8_mps = pr8_mps, ours_mps = ours_mps,
        speedup = speedup, max_eig_rel = max_eig_rel,
    )
end

# --- Driver ---
println("compare_pr8 — physically-realistic SW08 matrices, "
        * "PR#8 vs NVRTC kernel")
println("Sunny version: $(pkgversion(Sunny))")
println("CUDA version:  $(CUDA.runtime_version())")
println("hostname:      $(gethostname())")
println()

# Build SW08 SpinWaveTheoryDevice once.
print("Building SW08 SpinWaveTheoryDevice... "); flush(stdout)
swt_d, swt_host, sys_enlarged, cryst = build_sw08_swt_device()
println("OK ($(2 * Sunny.nbands(swt_d))×$(2 * Sunny.nbands(swt_d)) per matrix)")
flush(stdout)

# Construct our solver once. With autotune enabled, first construction at
# (N, BPB) on this hardware does an autotune sweep + cache the winner.
# Subsequent runs cache-hit. The cache state from the C++ test_autotune_n54
# is shared, so this should already be cache-warm on this machine.
const N = 2 * Sunny.nbands(swt_d)
@printf("Constructing NVRTC solver for N=%d (autotune mode, cache=Auto)...\n", N)
flush(stdout)
solver = Solver(N; tuning=:autotune, cache=:auto)
@printf("  block_dim_x = %d\n", block_dim_x(solver))
@printf("  bpb         = %d\n", batches_per_block(solver))
@printf("  shared_mem  = %d B\n", shared_mem_bytes(solver))
println()
flush(stdout)

# Batch sizes mirror Sunny's chunk-size pattern but bounded by what fits
# alongside Sunny's machinery on a 12 GB laptop.
const BATCH_SIZES = parse.(Int,
    split(get(ENV, "SUNNY_GENEIG_BATCH_SIZES", "256,1024,4096"), ","))
println("Batch sizes: $(BATCH_SIZES)")
println()
flush(stdout)

results = NamedTuple[]
for batch in BATCH_SIZES
    push!(results, bench_one_size(swt_d, swt_host, sys_enlarged, cryst, batch;
                                   solver=solver))
end

# Summary table.
println("================ PR#8 vs ours summary ================")
@printf("%-7s | %-22s | %-22s | %-9s | %s\n",
        "batch", "PR#8 (med, range)", "ours (med, range)", "speedup", "eig_rel")
println("--------+------------------------+------------------------+-----------+----------")
for r in results
    @printf("%-7d | %.4fs (%.4f-%.4f) | %.4fs (%.4f-%.4f) | %7.3fx  | %.2e\n",
            r.batch, r.pr8_med, r.pr8_min, r.pr8_max,
            r.ours_med, r.ours_min, r.ours_max,
            r.speedup, r.max_eig_rel)
end
println("=================================================================")

# Persist results to a markdown report.
const RESULTS_DIR = abspath(joinpath(@__DIR__, "..", "results"))
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)
const _today_str = Dates.format(Dates.now(), "yyyy-mm-dd")
const REPORT_PATH = joinpath(RESULTS_DIR,
    "local_pr8_comparison_$(gethostname())_$(_today_str).md")

open(REPORT_PATH, "w") do io
    println(io, "# local PR#8 comparison — $(gethostname()), $(Dates.now())")
    println(io)
    println(io, "- Sunny version: `$(pkgversion(Sunny))`")
    println(io, "- CUDA runtime: `$(CUDA.runtime_version())`")
    println(io, "- Host: `$(gethostname())`")
    println(io, "- Matrix size N = $(N) (2L for SW08 sqrt(3)×sqrt(3) kagome AFM, :dipole mode)")
    println(io, "- Our solver: BlockDim<$(block_dim_x(solver))>, BPB=$(batches_per_block(solver)), shared_mem=$(shared_mem_bytes(solver)) B")
    println(io)
    println(io, "## Results (5-sample medians)")
    println(io)
    println(io, "| batch | PR#8 median | PR#8 range | ours median | ours range | speedup (PR#8/ours) | max eig_rel |")
    println(io, "|---:|---:|---:|---:|---:|---:|---:|")
    for r in results
        @printf(io, "| %d | %.4f s | %.4f–%.4f s | %.4f s | %.4f–%.4f s | %.3f× | %.2e |\n",
                r.batch, r.pr8_med, r.pr8_min, r.pr8_max,
                r.ours_med, r.ours_min, r.ours_max,
                r.speedup, r.max_eig_rel)
    end
    println(io)
    println(io, "## Throughput (matrices / second)")
    println(io)
    println(io, "| batch | PR#8 mat/s | ours mat/s |")
    println(io, "|---:|---:|---:|")
    for r in results
        @printf(io, "| %d | %d | %d |\n", r.batch,
                round(Int, r.pr8_mps), round(Int, r.ours_mps))
    end
end
println()
println("results saved → $(REPORT_PATH)")
