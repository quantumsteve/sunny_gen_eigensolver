# Hybrid path — `HybridGeneigSolver`

NVRTC reduce-only kernel + cuSOLVER's `cusolverDnXsyevBatched` +
cuBLAS's `cublasZtrsmBatched_64` for the back-transform. Built to address
the load-bearing finding from per-stage profiling of the unified path:
**heev + back-trsm is 89% of the unified kernel's wall time, and
cuSOLVER's batched syev is 1.41× faster on the same workload at
N=54 batch=1024**.

## Architecture

```
HybridGeneigSolver::launch(d_H, d_S, d_W, d_U, d_info, batch, stream)
        │
        ├── stage 1: NVRTC reduce-only kernel (one cuLaunchKernel)
        │     potrf(S) → L                  [cuSolverDx::Cholesky in smem]
        │     trsm × 2: M := L⁻¹·H·L⁻ᴴ      [cuSolverDx::Trsm in smem]
        │     write L to L_buf_, M to d_U
        │
        ├── stage 2: cuSOLVER (one batched call → multiple kernels)
        │     cusolverDnXsyevBatched(handle, V, L, n, ComplexF64,
        │                              d_U, n, Float64, d_W, ComplexF64,
        │                              ws_gpu, ws_cpu, d_info, batch)
        │     overwrites d_U with eigenvectors V
        │     populates d_W with eigenvalues (ascending)
        │
        └── stage 3: cuBLAS (one batched call → one kernel)
              build pointer arrays L_ptrs, U_ptrs (cuMemcpyHtoDAsync)
              cublasZtrsmBatched_64('L','L','C','N', n, n, 1.0,
                                     L_ptrs, n, U_ptrs, n, batch)
              overwrites d_U with U := L⁻ᴴ·V
```

Source: [src/hybrid/reduce_kernel_source.hpp](../src/hybrid/reduce_kernel_source.hpp),
[src/hybrid/hybrid_solver.cpp](../src/hybrid/hybrid_solver.cpp).

All three stages submit to the user's stream; **0 host-blocking syncs**
inside `launch()`. PR#8's pipeline has 2 `findfirst(!iszero, info)`
host-side asserts; we elide them and return info as a raw output for
the caller to check asynchronously.

## Trade-off rationale

We pay:
- 2 extra inter-stage **global memory round-trips** (L spill at the end
  of stage 1, M re-read at the start of stage 2; V re-read between
  stages 2 and 3).
- 2 extra **kernel launches** (3 total vs unified's 1).
- 2 extra **`cuMemcpyHtoDAsync`** for pointer-array population.
- A **per-batch device workspace** of ~1.85 MB/matrix
  (`cusolverDnXsyevBatched`'s scratch). At B=4096 this hits 7.2 GB and
  exceeds the 12 GB laptop ceiling (OOM); at B=1024 it's 1.8 GB and
  fits comfortably.

In return:
- Stages 2 and 3 run **batch-axis-parallel** (cuBLAS/cuSOLVER kernels
  use grid-stride patterns over the batch dimension), which the
  unified kernel cannot do because its 99 KB per-block shared-memory
  footprint forces 1-block-per-matrix.
- Stage 1 keeps our shared-memory advantage on the cheap operators
  (potrf + trsm × 2 = 11% of unified wall time on this hardware).

## Validation history

- **test_hybrid_n54** ([tests/hybrid/test_hybrid_n54.cpp](../tests/hybrid/test_hybrid_n54.cpp)):
  8/8 PASS at fp64 epsilon on the n54_b8_s42_random LAPACK fixture.
  `max_eig_rel = 2.220 × 10⁻¹⁶` — actually **better than the unified
  path's 2.165 × 10⁻¹⁵**. cuSOLVER's batched syev follows LAPACK's QR
  convention more closely than cuSolverDx's heev; the residual is
  near machine epsilon.
- **test_hybrid_multiN** ([tests/hybrid/test_hybrid_multiN.cpp](../tests/hybrid/test_hybrid_multiN.cpp)):
  4/4 (N, fixture) pairs PASS at N ∈ {16, 32, 48, 54}, all eig_rel <
  1 × 10⁻¹⁵. Cold-compile time of the reduce-only kernel is **5–12 s**
  (vs 180–215 s for the unified kernel — fewer cuSolverDx primitives
  pulled into LTO IR ⇒ faster nvJitLink).

Cross-stack eigenvalue agreement against PR#8 stays at the same
~1 × 10⁻⁸ as the unified path measures (different rounding orders in
the iterative QR; not a regression).

## Operational characteristics

- First construction: ~10–15 s (reduce-only kernel NVRTC + nvJitLink;
  no autotune sweep, no cubin cache yet — ships caching for
  the unified path only).
- Per-launch: 1 NVRTC kernel + 1 cuSOLVER kernel-batch + 1 cuBLAS
  kernel + 2 `cuMemcpyHtoDAsync`. Total ~3 host-side calls; **0**
  host-blocking syncs.
- Workspace memory: `xsyev_ws_gpu_` ≈ 1.85 MB · batch_size, sized on
  first launch and reused across subsequent launches.

## Limitations

- **cuSOLVER `XsyevBatched` workspace ceiling.** At N=54 the workspace
  is ~1.85 MB per matrix. Combined with input/output buffers, this
  exceeds the 12 GB laptop ceiling at batch ≥ 4096. The hybrid path is
  unavailable for very large single batches on this hardware. 's chunked-launch primitive is a possible mitigation if we ever
  need it; H100's 80 GB pushes the ceiling to ~40k matrices.
- **No on-disk cubin cache** for the reduce-only kernel yet. Each
  construction recompiles. Adding 3.5f-style caching for the
  reduce-only source string is a future-work item.
- **No autotune for BlockDim.** The reduce-only kernel uses BlockDim<96>
  (the unified-path autotune-picked value). The reduce-only kernel has
  no heev to balance against; its own optimum may differ. Re-running
  the 3.5e sweep specifically for this kernel is a future-work item.

## Public API surface

C++:
```cpp
explicit HybridGeneigSolver(int n, int device_id = 0);
void launch(CUdeviceptr d_H, CUdeviceptr d_S,
            CUdeviceptr d_W, CUdeviceptr d_U, CUdeviceptr d_info,
            int batch_size, CUstream stream);
```

C ABI: `sunny_geneig_create_ex(..., kernel_mode=SUNNY_GENEIG_KERNEL_HYBRID)`.
Julia: `Solver(n; kernel=:hybrid)`.

## Related documents

- [UNIFIED_PATH.md](UNIFIED_PATH.md) — the unified single-kernel counterpart.
- [COMPARISON.md](COMPARISON.md) — three-way comparison methodology (this path
  vs the unified path vs Sunny.jl's PR#8).
