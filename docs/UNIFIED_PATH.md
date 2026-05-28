# Unified path — `UnifiedGeneigSolver`

Single-launch fused NVRTC kernel that runs the full generalized
eigenvalue pipeline `potrf → trsm × 2 → heev → trsm` inside one block per
matrix, all in shared memory. This is the architectural-novelty path: it
demonstrates that a fused multi-operator cuSolverDx kernel is buildable,
correct at fp64 epsilon across N ∈ {16, 32, 48, 54} on consumer
hardware, and competitive with library-call composition at small-to-mid
batch sizes. See [COMPARISON.md](COMPARISON.md) for measured throughput
vs PR#8 and the hybrid path.

## Architecture

```
UnifiedGeneigSolver::launch(d_H, d_S, d_W, d_U, d_info, batch, stream)
        │
        └── one cuLaunchKernel into geneig_full_kernel:
              cooperative load: H → Bs (smem), S → As (smem)
              __syncthreads
              Cholesky::execute(As, info_b)             // potrf in-place
              __syncthreads
              TrsmLeft::execute (As, Bs)                 // Bs := L⁻¹·H
              __syncthreads
              TrsmRight::execute(As, Bs)                 // Bs := Bs·L⁻ᴴ
              __syncthreads
              Heev::execute(Bs, lambda_s, workspace_s)  // overwrite-vec
              __syncthreads
              TrsmLeftConj::execute(As, Bs)              // back-transform
              __syncthreads
              cooperative store: lambda_s → W_b, Bs → U_b
```

Source: [src/unified/kernel_source.hpp](../src/unified/kernel_source.hpp),
[src/unified/unified_solver.cpp](../src/unified/unified_solver.cpp).

## Validation history

- **Static-link reference**: hardcoded N=54 device kernels in
  `src/geneig_fixed.cu` validated 8/8 PASS at fp64 epsilon against
  LAPACK on the random and near-identity fixtures
  ([tests/static/](../tests/static/)).
- **First NVRTC plumbing**: a hybrid two-kernel split was required to
  work around the cuSolverDx 25.12 NVRTC heev bug; see
  [CUSOLVERDX_NVRTC_HEEV_BUG.md](CUSOLVERDX_NVRTC_HEEV_BUG.md).
- **cuSolverDx header overlay applied**: the unified kernel now runs
  all five stages in one launch. Validates 8/8 (N, fixture) pairs
  across N ∈ {16, 32, 48, 54} (test_multiN). N ≥ 64 fails the
  cuSolverDx static_assert because `Cholesky::shared_memory_size +
  Heev::shared_memory_size` exceeds the sm_120 device's 101 KB
  per-block ceiling.

## Optimization sweeps

| Knob | Outcome |
|---|---|
| `BatchesPerBlock` auto-select | BPB=1 at N=54 (smem ceiling); BPB=8 at N=16, BPB=2 at N=32, BPB=1 at N=48/54. |
| Per-(N, BPB, arch) `BlockDim.x` from cuSolverDx's `suggested_block_dim` | At N=32 BPB=2: BlockDim<64> (suggested) is 1.153× faster than BlockDim<128> (hardcode). At N=54 BPB=1 the suggested 128 matches the prior hardcode. |
| Inter-primitive sync minimization | Source-level analysis suggested 3 of our 5 syncs were redundant. Empirically removing them **regressed** throughput by 14% at N=54 BPB=1 (cuSolverDx's internal sync only covers participating warps). Documented null result; we keep all 5 syncs. |
| Multi-stream chunked launch | No throughput improvement on this hardware: the kernel already saturates the GPU at modest batches; adding streams contributes only host-side overhead. Documented null result. |
| Empirical autotune around suggested BlockDim | **At N=54 BPB=1: 1.198× speedup**. Autotune picked BlockDim=96 (suggested was 128). cuSolverDx's max-across-operators heuristic over-shoots when one operator (heev's 128) dominates the others (chol/trsm at 64). See [CUSOLVERDX_FUSED_BLOCKDIM_FINDING.md](CUSOLVERDX_FUSED_BLOCKDIM_FINDING.md). |
| Two-layer module cache (in-process + on-disk) | 213 s cold compile → 285 ms warm-disk → 0.007 ms in-process. ELF-magic + size validation; atomic rename for multi-process safety. |

## Known characterization

The per-stage diagnostic (test_heev_fraction) found:

```
full       (potrf+trsm×2+heev_overwrite_vec+back-trsm) :  0.12661 s   100.0%
reduce-only (potrf+trsm×2)                              :  0.01388 s    11.0%
no-vectors (potrf+trsm×2 + heev no-vec)                 :  0.08058 s    63.6%
```

**Heev + back-trsm is 89% of unified-kernel wall time at N=54, BPB=1.**
This is the load-bearing finding that motivated the hybrid path
([HYBRID_PATH.md](HYBRID_PATH.md)).

## Limitations

- **Shared-memory ceiling at N=54.** `Cholesky::shared_memory_size +
  Heev::shared_memory_size = 99,712 B` at N=54 BPB=1 essentially fills
  the sm_120 laptop's 101 KB per-block opt-in dynamic shared memory.
  This forces BPB=1 (one matrix per block resident on each SM), which
  caps cross-batch parallelism. H100's 228 KB ceiling should let
  BPB=2 fit; whether heev's per-block warp-driver actually exploits it
  is a measurement.
- **Heev iterative QR is single-warp.** Inside cuSolverDx's
  `htev::dispatch`, the iterative QR runs on one warp per matrix even
  with BPB > 1. Cross-batch parallelism comes from multiple matrices
  per block, not from spreading one matrix across more threads.

## Operational characteristics

- First-ever construction: ~213 s NVRTC + nvJitLink at N=54 BPB=1
  BlockDim=128 (the original probe-suggested value); ~210 s for the
  3.5e autotune sweep's winner BlockDim=96.
- Repeat construction (cached): 285 ms warm-disk, 0.007 ms in-process.
- Per-launch: 1 `cuLaunchKernel`; **0** host-blocking syncs.

## Public API surface

C++:
```cpp
explicit UnifiedGeneigSolver(int n, int device_id,
                              int batches_per_block_request = 0,
                              int force_block_dim_x = 0,
                              TuningMode tuning_mode = TuningMode::Suggested,
                              CacheMode  cache_mode  = CacheMode::Auto);
void launch(CUdeviceptr d_H, CUdeviceptr d_S,
            CUdeviceptr d_W, CUdeviceptr d_U, CUdeviceptr d_info,
            int batch_size, CUstream stream);
void launch_chunked(...);  // 3.5d round-robin chunked launch
```

C ABI: `sunny_geneig_create_ex(..., kernel_mode=SUNNY_GENEIG_KERNEL_UNIFIED)`.
Julia: `Solver(n; kernel=:unified)`.

## Related documents

- [NVRTC_RUNTIME.md](NVRTC_RUNTIME.md) — detailed development log for the
  NVRTC pipeline (autotune, BPB selection, cubin cache, multi-stream).
- [HYBRID_PATH.md](HYBRID_PATH.md) — the hybrid three-call counterpart.
- [COMPARISON.md](COMPARISON.md) — three-way comparison methodology.
