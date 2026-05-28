# Fixed-N=54 In-Kernel Batched Generalized Eigenvalue Solver

Implements `H u = λ S u` for complex Hermitian-definite (H, S) pairs entirely
inside a single CUDA kernel using cuSolverDx primitives composed in shared
memory. Built incrementally across five sub-steps, each independently
validated against LAPACK at fp64 epsilon.

## Pipeline

Each block stages H, S into shared memory, then composes:
1. `cusolverdx::potrf` — Cholesky factor S = L Lᴴ in place
2. `cusolverdx::trsm(left, non_trans)` — apply L⁻¹ from the left to H
3. `cusolverdx::trsm(right, conj_trans)` — apply L⁻ᴴ from the right
   Result is M = L⁻¹ H L⁻ᴴ, the standard-form reduction
4. `cusolverdx::heev(overwrite_vectors)` — eigendecompose M, returning
   eigenvalues Λ and eigenvectors V of the reduced problem
5. `cusolverdx::trsm(left, conj_trans)` — back-transform U = L⁻ᴴ V to
   recover the generalized eigenvectors

The key win is shared-memory persistence across primitives: As (the L
factor) stays resident through all five operations; Bs is reused for
H → M → V → U.

## Sub-step validation history

| Sub-step | Test target | Reference | Max rel diff |
|---|---|---|---|
| 2.2a Cholesky | test_chol_n54 | LAPACKE_zpotrf | 7.6e-17 |
| 2.2b Reduction (potrf+2×trsm) | test_reduce_n54 | LAPACKE_zhegst | 1.4e-16 |
| 2.2c Reduction + eigenvalues | test_reduce_eig_n54 | LAPACKE_zheev | 2.4e-15 (eig) / 5.1e-15 (phase) |
| 2.2d Full generalized | test_full_n54 | LAPACKE_zhegv | 5.7e-15 (eig) / 3.8e-15 (phase) |
| 2.2e Batched dispatch | test_full_batched_n54 | LAPACKE_zhegv ×1024 | 3.1e-15 / 6.0e-15 |

All metrics are single-digit ulps at fp64. The criterion was 1e-10 (eig)
and 1e-10 (phase) for sub-steps c-e and 1e-12 for a-b; we cleared by 5+
orders of magnitude in every case.

## Resources

- Shared memory per block: 99,712 bytes
  - As tile (S→L): 46,656 bytes
  - Bs+lambda+workspace (heev region): 53,056 bytes
- Block dimension: 128 threads
- Targets compute capability sm_70+ via cuSolverDx fatbin selection;
  development verified on sm_120 (RTX 5070 Ti laptop, Blackwell)

## Performance reference (development hardware, 5070 Ti laptop, Blackwell)

On 1024 random Hermitian-definite 54×54 pairs:
- Our kernel: ~0.15 s (6,680 matrices/sec)
- LAPACK zhegv (single-threaded host reference): ~0.87 s (1,183 matrices/sec)
- Ratio: ~5.6×

This is a correctness-validated reference implementation, not a tuned
benchmark. Performance comparisons against MAIQMag/Sunny.jl PR #8
(batched cuSOLVER path) are deferred to with the H100 hardware
that matches the SEED-project email thread baseline.

## What this implements that didn't exist before

cuSolverDx 25.12 ships primitives for Cholesky, trsm, and Hermitian
eigendecomposition individually but no fused composition for the
complex Hermitian-definite generalized eigenvalue problem. Sunny.jl's
PR #8 path uses a chain of cusolver/cublas batched library calls, which
limits per-call kernel launch overhead and cross-primitive memory
traffic. This kernel demonstrates the in-kernel composition pattern
Simon Byrne suggested in the NVIDIA/AmSC SEED-project email thread
(Feb 2 2026) for matrix sizes where shared-memory persistence is
feasible.

## Next: (NVRTC)

The N=54 size is hardcoded in the cuSolverDx solver-type templates.
Sunny's matrix size is determined at runtime by the user's magnetic
unit cell (twice the number of atoms × local Hilbert space dimension),
so a fixed-size kernel is not a usable artifact for Sunny integration.
templates the kernel source on N and uses NVRTC to compile
each requested size on demand, with a per-size CUmodule cache.
