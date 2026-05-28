# cuSolverDx 25.12: per-operator `suggested_block_dim` is suboptimal for fused kernels

## Summary

`suggested_block_dim` is computed per operator under the implicit assumption
of single-operator kernels. When several operators are fused into one kernel
launched with one `BlockDim`, taking `max(suggested_block_dim.x)` across the
operators can land on a value where the under-utilized operators
collectively cost more wall time than the single dominant operator gains
from running at its preferred width. Empirical autotuning routinely picks a
smaller `BlockDim.x` than the per-operator max and produces measurable
speedup at fp64 epsilon.

## Reproducer

The unified five-stage generalized eigenvalue kernel in this repo fuses
`Cholesky::potrf` + `Trsm × 3` + `Heev` (the standard reduction-to-standard-
form pipeline `potrf → trsm × 2 → heev → back-trsm`) inside one launch.
At N=54, BPB=1, on an RTX 5070 Ti Laptop GPU (sm_120, CUDA 12.8), the
per-operator `suggested_block_dim.x` values cuSolverDx returns are:

| Operator                  | suggested_block_dim.x |
|---------------------------|----------------------:|
| `Cholesky::potrf`         |                    64 |
| `Trsm` (left, non_trans)  |                    64 |
| `Trsm` (right, conj_trans)|                    64 |
| `Trsm` (left, conj_trans) |                    64 |
| `Heev`                    |                   128 |

The straightforward composition rule — take the max so every operator gets
at least its required width — gives 128. Empirical autotuning picks 96.

## Measured impact

5-sample median throughput on 1024 random Hermitian-definite 54×54 matrix
pairs, NVRTC build, same workload across all rows:

| BlockDim.x      | mat/sec | speedup |
|-----------------|--------:|--------:|
| 128 (max)       |   6,757 |  1.000× |
| **96 (autotune)** | **8,092** | **1.198×** |
|  64             |   6,286 |  0.930× |

The autotuned 96 outruns the max-rule 128 by 19.8%. `BlockDim.x < N` is not
generally feasible for the trsm operators here, which sets the lower bound
on the candidate set.

## Mechanism

At `BlockDim<128>`, `Cholesky::potrf` and the three `Trsm` operators run
with twice the threads they prefer. The extra threads inflate register
pressure and warp-scheduling overhead on every block. `Heev` does want 128
threads in its single-operator suggestion, but at N=54 its working-set
distribution doesn't fully utilize that width either — 96 is enough to keep
heev at peak while leaving the four 64-want operators much closer to their
per-operator optimum. Across the five fused stages the trade lands in
favour of the smaller block. The 64 row in the table above shows that
going *below* heev's threshold pays the opposite cost: heev under-runs and
drags the whole pipeline down.

The N=32, BPB=2 configuration in this project's autotune test does *not*
show the same effect — there the per-operator suggestions are
`(64, 64, 64, 64, 32)`, the max is 64 by majority not by heev, and autotune
correctly converges back to 64. The over-shoot only appears when one
operator's suggestion is strictly larger than the rest and the fused-kernel
share of that operator's wall time is small relative to the rest.

## Implication

For fused multi-operator cuSolverDx kernels at small to medium N,
`suggested_block_dim` is a reasonable starting point but should be treated
as an upper bound to sweep down from, not a final value. The model behind
the recommendation implicitly assumes the operator owns the kernel.

A natural cuSolverDx enhancement: a `fused_suggested_block_dim<Op1, Op2,
...>` template that takes the operator-type list and returns a value tuned
for the fused case. The empirical traces in this project's
[`tests/unified/test_autotune_n54.cpp`](../tests/unified/test_autotune_n54.cpp) and
[`tests/unified/test_autotune_n32.cpp`](../tests/unified/test_autotune_n32.cpp) — winning
`BlockDim.x` per (N, BPB, arch) plus the per-candidate sweep tables — could
seed such a model.

## Reference

- Implementation: [`src/unified/unified_solver.cpp`](../src/unified/unified_solver.cpp),
  `autotune_block_dim_x_` and `TuningMode::Autotune`.
- Tests: [`tests/unified/test_autotune_n54.cpp`](../tests/unified/test_autotune_n54.cpp),
  [`tests/unified/test_autotune_n32.cpp`](../tests/unified/test_autotune_n32.cpp).
- Phase log: [`docs/NVRTC_RUNTIME.md`](NVRTC_RUNTIME.md), section 3.5e
  (per-candidate sweep tables, the heuristic-over-shoot diagnosis, and the
  bring-up bugs found during 3.5e).
- Discovered during a benchmark study of in-kernel cuSolverDx composition
  vs cuSOLVER batched library calls (NVIDIA / AmSC SEED collaboration with
  Oak Ridge; project goal: GPU-side generalized eigenvalue solver for the
  Sunny.jl spin-dynamics package).
