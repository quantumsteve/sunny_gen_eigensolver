# reference baseline

Reproduction of MAIQMag/Sunny.jl `SW08_sqrt3_kagome_AFM_CUDA.jl` on the development machine.

## Hardware
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, sm_120, 12 GB)
- Driver: 572.97 (CUDA 12.8)
- Host: WSL2 Ubuntu, CUDA Toolkit 12.8, Julia 1.12.6
- CUDA.jl 5.11.2, artifact-managed runtime CUDA 12.9

## Workload (per `SW08_sqrt3_kagome_AFM_CUDA.jl` upstream)
- 27-site magnetic cell (3-site kagome × √3×√3×1 spiral enlargement)
- Hermitian-definite generalized eigenvalue problem, dimension 54×54, complex
- 200 × 200 = 40,000 q-points per `powder_average` call
- 600 × 600 = 360,000 q-points for the Steven-thread reference variant

## Wall-clock results

| Phase | Time (s) | Note |
|---|---|---|
| Warmup (40k, batch=3, cold) | 51.30 | Includes JIT compilation |
| Sweep batch_size=1 (40k) | 4.89 | Hot |
| Sweep batch_size=2 (40k) | 4.16 | Hot |
| Sweep batch_size=3 (40k) | 4.00 | Hot |
| Steven reference (360k, batch=3) | 34.26 | Hot |

## NVTX dominance
`geneig_batch` accounts for 45.7% of total wall time across 639 invocations (avg 149.7 ms/inst). This confirms the hot path identified in the NVIDIA/AmSC SEED-project email thread (Steven Hahn, Jan 30 2026): "the generalized eigenvalue solve via cusolver/cublas-batched calls dominates the runtime."

## Reference for downstream comparisons

The 360k-q-point variant (34.26s on 5070 Ti laptop) is the apples-to-apples reference for our cuSolverDx implementation. Steven's reported H100 NVL number for the same configuration is approximately 2.2s.
