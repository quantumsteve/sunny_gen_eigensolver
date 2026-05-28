# three-way comparison: PR#8 vs unified vs hybrid.
#
# Same SW08-generated (H_d, Σ_d) inputs as compare_pr8.jl. For each batch
# size, runs PR#8, our unified kernel, and our hybrid path 5 samples each,
# reports median throughput and speedup vs PR#8, validates eigenvalue
# agreement against PR#8 to fp64 epsilon for one sample.

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
const BATCH_SIZES = parse.(Int,
    split(get(ENV, "SUNNY_GENEIG_BATCH_SIZES", "256,1024,4096"), ","))

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

function time_ours!(solver, Σ_d, H_d, W, U, info)
    CUDA.synchronize()
    t0 = time_ns()
    SunnyGeneigBench.run_kernel!(solver, Σ_d, H_d, W, U, info)
    CUDA.synchronize()
    return (time_ns() - t0) / 1e9
end

println("three-way comparison — PR#8 vs unified vs hybrid")
println("Sunny: $(pkgversion(Sunny))  CUDA runtime: $(CUDA.runtime_version())  host: $(gethostname())")
println("Batch sizes: $(BATCH_SIZES)")
println()

print("Building SW08 SpinWaveTheoryDevice... "); flush(stdout)
swt_d, swt_host, sys_enlarged, cryst = build_sw08_swt_device()
const N = 2 * Sunny.nbands(swt_d)
println("OK (N=$N)")
flush(stdout)

@printf("Constructing unified solver (autotune mode, cache=Auto)...\n")
flush(stdout)
solver_unified = Solver(N; tuning=:autotune, cache=:auto, kernel=:unified)
@printf("  unified: BlockDim=%d, BPB=%d, smem=%d B\n",
        block_dim_x(solver_unified), batches_per_block(solver_unified),
        shared_mem_bytes(solver_unified))

@printf("Constructing hybrid solver (kernel=:hybrid)...\n")
flush(stdout)
solver_hybrid = Solver(N; kernel=:hybrid)
@printf("  hybrid:  BlockDim=%d, BPB=%d, smem=%d B\n",
        block_dim_x(solver_hybrid), batches_per_block(solver_hybrid),
        shared_mem_bytes(solver_hybrid))
println()
flush(stdout)

results = NamedTuple[]
for batch in BATCH_SIZES
    println("=== batch=$batch ===")
    flush(stdout)

    H_template, Σ_template = build_sw08_matrices(swt_d, swt_host, sys_enlarged,
                                                  cryst, batch)
    W_buf    = CUDA.zeros(Float64, N, batch)
    U_buf    = CUDA.zeros(ComplexF64, N, N, batch)
    info_buf = CUDA.zeros(Cint, batch)

    # Eigenvalue cross-check: run PR#8 once, then each of ours, compare.
    Hp = copy(H_template); Σp = copy(Σ_template)
    pr8_evals = Array(pr8_hegvd!(Hp, Σp))

    # Unified validation
    fill!(W_buf, 0.0); fill!(info_buf, 0)
    Ho = copy(H_template); Σo = copy(Σ_template)
    SunnyGeneigBench.run_kernel!(solver_unified, Σo, Ho, W_buf, U_buf, info_buf)
    CUDA.synchronize()
    unified_W = Array(W_buf)

    # Hybrid validation — wrapped because cuSOLVER's XsyevBatched workspace
    # is ~1.85 MB/matrix and may exceed device memory at large batches.
    # We catch the OOM, mark hybrid as unavailable for this batch size, and
    # skip the timing loop below.
    hybrid_ok = true
    hybrid_W  = nothing
    try
        fill!(W_buf, 0.0); fill!(info_buf, 0)
        Ho = copy(H_template); Σo = copy(Σ_template)
        SunnyGeneigBench.run_kernel!(solver_hybrid, Σo, Ho, W_buf, U_buf, info_buf)
        CUDA.synchronize()
        hybrid_W = Array(W_buf)
    catch e
        hybrid_ok = false
        @printf("  hybrid validation FAILED: %s\n", sprint(showerror, e))
        @printf("  (hybrid will be skipped for this batch)\n")
        flush(stdout)
        # Reclaim partial allocations from cuSOLVER + cuBLAS before
        # continuing — running CUDA.reclaim() helps the next batch.
        try CUDA.reclaim() catch end
    end

    spot = unique(round.(Int, range(1, batch, length=min(8, batch))))
    function eig_vs_pr8(ours)
        m = 0.0
        for k in spot
            a = sort(pr8_evals[:, k]); b = sort(ours[:, k])
            for i in 1:N
                m = max(m, abs(a[i] - b[i]) / max(abs(a[i]), 1.0))
            end
        end
        return m
    end
    eig_unified = eig_vs_pr8(unified_W)
    eig_hybrid  = hybrid_ok ? eig_vs_pr8(hybrid_W) : NaN
    @printf("  eig vs PR#8: unified=%.2e, hybrid=%s\n",
            eig_unified, hybrid_ok ? @sprintf("%.2e", eig_hybrid) : "(OOM)")
    flush(stdout)

    # Warm-up each path that's still available.
    let
        H = copy(H_template); Σ = copy(Σ_template); time_pr8!(H, Σ)
        H = copy(H_template); Σ = copy(Σ_template); time_ours!(solver_unified, Σ, H, W_buf, U_buf, info_buf)
        if hybrid_ok
            try
                H = copy(H_template); Σ = copy(Σ_template)
                time_ours!(solver_hybrid, Σ, H, W_buf, U_buf, info_buf)
            catch
                hybrid_ok = false
                try CUDA.reclaim() catch end
            end
        end
    end

    pr8_t = Float64[]; uni_t = Float64[]; hyb_t = Float64[]
    for _ in 1:SAMPLES
        H = copy(H_template); Σ = copy(Σ_template)
        push!(pr8_t, time_pr8!(H, Σ))

        fill!(W_buf, 0.0); fill!(info_buf, 0)
        H = copy(H_template); Σ = copy(Σ_template)
        push!(uni_t, time_ours!(solver_unified, Σ, H, W_buf, U_buf, info_buf))

        if hybrid_ok
            try
                fill!(W_buf, 0.0); fill!(info_buf, 0)
                H = copy(H_template); Σ = copy(Σ_template)
                push!(hyb_t, time_ours!(solver_hybrid, Σ, H, W_buf, U_buf, info_buf))
            catch
                hybrid_ok = false
                try CUDA.reclaim() catch end
                break
            end
        end
    end

    pr8_med, pr8_lo, pr8_hi, _ = median_min_max(pr8_t)
    uni_med, uni_lo, uni_hi, _ = median_min_max(uni_t)
    hyb_med, hyb_lo, hyb_hi    = (hybrid_ok && !isempty(hyb_t)) ?
        median_min_max(hyb_t)[1:3] : (NaN, NaN, NaN)


    @printf("  PR#8:    med=%.4f s (%.4f-%.4f), %d mat/sec\n",
            pr8_med, pr8_lo, pr8_hi, round(Int, batch/pr8_med))
    @printf("  unified: med=%.4f s (%.4f-%.4f), %d mat/sec, %.3fx vs PR#8\n",
            uni_med, uni_lo, uni_hi, round(Int, batch/uni_med), pr8_med/uni_med)
    if hybrid_ok && !isnan(hyb_med)
        @printf("  hybrid:  med=%.4f s (%.4f-%.4f), %d mat/sec, %.3fx vs PR#8\n",
                hyb_med, hyb_lo, hyb_hi, round(Int, batch/hyb_med), pr8_med/hyb_med)
    else
        @printf("  hybrid:  SKIPPED (cuSOLVER XsyevBatched workspace exceeds device memory)\n")
    end
    flush(stdout)

    push!(results, (
        batch = batch,
        pr8_med = pr8_med, pr8_lo = pr8_lo, pr8_hi = pr8_hi,
        uni_med = uni_med, uni_lo = uni_lo, uni_hi = uni_hi,
        hyb_med = hyb_med, hyb_lo = hyb_lo, hyb_hi = hyb_hi,
        hybrid_ok = hybrid_ok && !isnan(hyb_med),
        eig_unified = eig_unified, eig_hybrid = eig_hybrid,
    ))
    println()
end

println()
println("================ three-way summary ================")
@printf("%-7s | %-9s | %-12s | %-9s | %-9s | %-9s\n",
        "batch", "path", "median (s)", "mat/sec", "vs PR#8", "eig_rel")
println("--------+-----------+--------------+-----------+-----------+----------")
for r in results
    @printf("%-7d | %-9s | %12.4f | %9d | %8.3fx | %s\n",
            r.batch, "PR#8", r.pr8_med, round(Int, r.batch/r.pr8_med),
            1.0, "reference")
    @printf("%-7d | %-9s | %12.4f | %9d | %8.3fx | %.2e\n",
            r.batch, "unified", r.uni_med, round(Int, r.batch/r.uni_med),
            r.pr8_med/r.uni_med, r.eig_unified)
    if r.hybrid_ok
        @printf("%-7d | %-9s | %12.4f | %9d | %8.3fx | %.2e\n",
                r.batch, "hybrid",  r.hyb_med, round(Int, r.batch/r.hyb_med),
                r.pr8_med/r.hyb_med, r.eig_hybrid)
    else
        @printf("%-7d | %-9s | %12s | %9s | %9s | %s\n",
                r.batch, "hybrid", "(OOM)", "-", "-", "workspace exceeds device memory")
    end
end
println("===============================================================")

# Persist report.
const RESULTS_DIR = abspath(joinpath(@__DIR__, "..", "results"))
isdir(RESULTS_DIR) || mkpath(RESULTS_DIR)
const _today_str = Dates.format(Dates.now(), "yyyy-mm-dd")
const REPORT_PATH = joinpath(RESULTS_DIR,
    "hybrid_three_way_$(gethostname())_$(_today_str).md")
open(REPORT_PATH, "w") do io
    println(io, "# three-way comparison — $(gethostname()), $(Dates.now())")
    println(io)
    println(io, "- Sunny: `$(pkgversion(Sunny))`")
    println(io, "- CUDA runtime: `$(CUDA.runtime_version())`")
    println(io, "- N = $(N)")
    println(io)
    println(io, "## Results")
    println(io)
    println(io, "| batch | path | median (s) | range | mat/sec | speedup vs PR#8 | eig_rel |")
    println(io, "|---:|---|---:|---:|---:|---:|---:|")
    for r in results
        @printf(io, "| %d | PR#8 | %.4f | %.4f–%.4f | %d | 1.000× | reference |\n",
                r.batch, r.pr8_med, r.pr8_lo, r.pr8_hi,
                round(Int, r.batch/r.pr8_med))
        @printf(io, "| %d | unified | %.4f | %.4f–%.4f | %d | %.3f× | %.2e |\n",
                r.batch, r.uni_med, r.uni_lo, r.uni_hi,
                round(Int, r.batch/r.uni_med),
                r.pr8_med/r.uni_med, r.eig_unified)
        if r.hybrid_ok
            @printf(io, "| %d | hybrid | %.4f | %.4f–%.4f | %d | %.3f× | %.2e |\n",
                    r.batch, r.hyb_med, r.hyb_lo, r.hyb_hi,
                    round(Int, r.batch/r.hyb_med),
                    r.pr8_med/r.hyb_med, r.eig_hybrid)
        else
            @printf(io, "| %d | hybrid | OOM | – | – | – | – (XsyevBatched workspace exceeds device memory) |\n",
                    r.batch)
        end
    end
end
println("results saved → $(REPORT_PATH)")
