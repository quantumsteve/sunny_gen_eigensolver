# sunny_gen_eigensolver

A batched complex Hermitian-definite generalized eigenvalue solver
(`H u = λ S u`) for GPU, built on cuSolverDx + NVRTC. Targets the
spin-wave dispersion workload of [Sunny.jl](https://github.com/SunnySuite/Sunny.jl)
on neutron-scattering-scale magnon problems. Developed as part of the
NVIDIA / Oak Ridge AmSC SEED collaboration.

The repository ships two distinct implementations of the same problem,
sharing an NVRTC compile/cache layer and a C ABI:

| Path | What it does | Strengths |
|---|---|---|
| **Unified** (`UnifiedGeneigSolver`) | Single fused NVRTC kernel: `potrf → trsm × 2 → heev → trsm` in shared memory, one block per matrix. | Demonstrates that a multi-operator cuSolverDx kernel is buildable, correct at fp64 epsilon, and competitive at small/mid batches. Architecturally novel. |
| **Hybrid** (`HybridGeneigSolver`) | NVRTC reduce-only kernel + `cusolverDnXsyevBatched` + `cublasZtrsmBatched_64`. | 1.41× faster heev stage on the laptop; matches Sunny.jl PR#8 throughput within run-to-run noise at small/mid batches. |

Both paths are exposed through a stable C ABI (`libsunny_geneig.so`) and a
Julia FFI wrapper (`julia/SunnyGeneigBench/`), with switching at solver
construction time.

## Findings of interest to NVIDIA

This work uncovered two cuSolverDx 25.12 behaviors that may matter to the
cuSolverDx team. Both are documented here with reproducers in `tests/`.

- **`function::heev` fails to instantiate under NVRTC** —
  [`docs/CUSOLVERDX_NVRTC_HEEV_BUG.md`](docs/CUSOLVERDX_NVRTC_HEEV_BUG.md).
  Root cause is an in-class-body name-lookup ordering issue between
  `workspace_size` and `max_threads_per_block`. Worked around in this
  repo via a one-line header overlay
  ([`src/common/cusolverdx_overlay/`](src/common/cusolverdx_overlay/)).

- **Per-operator `suggested_block_dim` is suboptimal for fused kernels** —
  [`docs/CUSOLVERDX_FUSED_BLOCKDIM_FINDING.md`](docs/CUSOLVERDX_FUSED_BLOCKDIM_FINDING.md).
  Taking `max(suggested_block_dim.x)` across composed operators
  over-shoots when one operator dominates. Empirical autotune at N=54
  picks BlockDim=96 over the suggested 128 for a measured 1.198×
  speedup at fp64 epsilon.

## Prerequisites

- CUDA Toolkit 12.6.3+ (12.8 confirmed)
- NVIDIA MathDx 25.12.1 (cuSolverDx + bundled CUTLASS)
- NVIDIA GPU with compute capability sm_70 or newer (sm_120 / Blackwell confirmed)
- CMake 3.24+
- LAPACK / LAPACKE (for reference fixtures and benchmark validation)

Set `MATHDX_ROOT` to the MathDx install root (the directory containing
`nvidia/mathdx/25.12/`). The provided
[`scripts/env.sh`](scripts/env.sh) defaults it to
`$HOME/opt/nvidia-mathdx-25.12.1-cuda12`.

## Build

```
source scripts/env.sh
cmake -B build -S .
cmake --build build -j
```

## Smoke test

```
./build/src/smoke
```

Expected output:

```
cuSolverDx smoke OK, shared mem bytes: 8448
```

## Where to start reading

- [`docs/UNIFIED_PATH.md`](docs/UNIFIED_PATH.md) — architecture of the
  single-launch fused-kernel path, validation history, optimization
  sweeps, and known characterization (the heev ceiling).
- [`docs/HYBRID_PATH.md`](docs/HYBRID_PATH.md) — architecture of the
  three-call hybrid path and its trade-offs.
- [`docs/COMPARISON.md`](docs/COMPARISON.md) — three-way methodology
  comparing the unified path, the hybrid path, and Sunny.jl's PR#8.
- [`docs/NVRTC_RUNTIME.md`](docs/NVRTC_RUNTIME.md) — detailed development
  log: NVRTC plumbing, BlockDim autotune, cubin cache, multi-stream
  experiment, BatchesPerBlock auto-select.
- [`docs/KERNEL_ARCHITECTURE.md`](docs/KERNEL_ARCHITECTURE.md) — the
  static-link reference kernel that the NVRTC path was validated against.

## Repository layout

```
src/
  common/      shared infrastructure: NVRTC plumbing, C ABI, cuSolverDx overlay
  unified/     UnifiedGeneigSolver + 5-stage fused kernel source
  hybrid/      HybridGeneigSolver + reduce-only kernel source
  geneig_fixed.cu, smoke.cu — static-link reference kernels

tests/
  static/      Fixed-N static-link reference tests (validate against LAPACK)
  common/      Path-agnostic infrastructure tests (C ABI, NVRTC repros, benchmarks)
  unified/     UnifiedGeneigSolver path tests
  hybrid/      HybridGeneigSolver path tests
  data/        Pre-generated LAPACK fixtures (binary)

docs/          Architecture, NVIDIA findings, comparison methodology
cmake/         Find-modules for MathDx and LAPACKE
scripts/       env.sh and smoke helpers
benchmarks/    SW08 baseline workload (kagome AFM, the SEED-project hot path)
julia/         Julia FFI wrapper + PR#8 comparison scripts
```

## Public API

C ABI (see [`src/common/c_api.h`](src/common/c_api.h)):

```c
sunny_geneig_solver_t sunny_geneig_create_ex(int n, int device_id,
                                              sunny_geneig_kernel_mode_t mode);
sunny_geneig_status_t  sunny_geneig_launch  (sunny_geneig_solver_t s,
                                              uintptr_t d_H, uintptr_t d_S,
                                              uintptr_t d_W, uintptr_t d_U,
                                              uintptr_t d_info,
                                              int batch_size, void* stream);
void                   sunny_geneig_destroy (sunny_geneig_solver_t s);
```

Julia wrapper (see
[`julia/SunnyGeneigBench/src/SunnyGeneigBench.jl`](julia/SunnyGeneigBench/src/SunnyGeneigBench.jl)):

```julia
solver = Solver(n; kernel=:unified)   # or :hybrid
run_kernel!(solver, H_d, S_d, W_d, U_d, info_d, batch)
```

## License

See [`LICENSE`](LICENSE).
