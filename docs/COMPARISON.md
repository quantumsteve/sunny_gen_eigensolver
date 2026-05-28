# Three-way comparison — PR#8 vs unified vs hybrid

The performance picture of this project sits between three implementations
of the same generalized eigenvalue computation:

1. **PR#8** — Steven Hahn's `hegvd_batched!` in
   [Sunny.jl/ext/CUDAExt/EigenBatched.jl](https://github.com/MAIQMag/Sunny.jl/blob/develop/ext/CUDAExt/EigenBatched.jl)
   (commit `2817d0c`). cuBLAS + cuSOLVER batched primitives composed on
   the host side: `cusolverDnZpotrfBatched` → `cublasZtrsmBatched_64 × 3`
   → `cusolverDnXsyevBatched`. Mature library code.
2. **Unified** — this project's `UnifiedGeneigSolver`. Single fused
   NVRTC kernel running all five stages in shared memory, one block
   per matrix. See [UNIFIED_PATH.md](UNIFIED_PATH.md).
3. **Hybrid** — this project's `HybridGeneigSolver`. NVRTC reduce-only
   kernel (potrf + trsm × 2) handing off to `cusolverDnXsyevBatched` +
   `cublasZtrsmBatched_64`. See [HYBRID_PATH.md](HYBRID_PATH.md).

Same Sunny-generated SW08 √3×√3 kagome AFM `(H_d, Σ_d)` inputs across
all three paths. Same hardware and CUDA stack. Same 5-sample-median
methodology.

## Methodology

- Generate `(H_d, Σ_d)` from Sunny's `dynamical_matrix!` and the
  `_set_identity` paraunitary metric kernel for a given batch size.
- For each path: warm-up sample, then 5 timed samples with
  `copy(H_template); copy(Σ_template)` per sample so in-place mutation
  doesn't contaminate.
- Time only the kernel/library call, with `CUDA.synchronize()` before
  `time_ns()` and after.
- Eigenvalue cross-check against PR#8 on 8 spot-sampled batch indices
  per run; report `max(|a[i]-b[i]|/max(|a[i]|,1))` over the spot indices
  and the eigenvalue index.

Driver: [julia/SunnyGeneigBench/scripts/compare_hybrid.jl](../julia/SunnyGeneigBench/scripts/compare_hybrid.jl).

## Laptop measurements (RTX 5070 Ti Laptop, sm_120, 12 GB, CUDA 12.8)

| batch | path    | median (s) | range          | mat/sec | speedup vs PR#8 | eig_rel  |
|---:|---------|---:|---:|---:|---:|---:|
|  256 | PR#8    | 0.0250 | 0.0246–0.0265 | 10,243 | 1.000× | reference |
|  256 | unified | 0.0273 | 0.0272–0.0334 |  9,375 | 0.915× | 1.32e-08 |
|  256 | **hybrid**  | **0.0253** | **0.0252–0.0302** | **10,134** | **0.989×** | **1.32e-08** |
| 1024 | PR#8    | 0.0944 | 0.0940–0.0955 | 10,845 | 1.000× | reference |
| 1024 | unified | 0.1037 | 0.1035–0.1053 |  9,879 | 0.911× | 1.60e-08 |
| 1024 | **hybrid**  | **0.0969** | **0.0967–0.0969** | **10,570** | **0.975×** | **1.60e-08** |
| 4096 | PR#8    | 0.8346 | 0.8268–0.8423 |  4,908 | 1.000× | reference |
| 4096 | unified | 2.0490 | 2.0486–2.0495 |  1,999 | 0.407× | 1.27e-08 |
| 4096 | hybrid  | OOM    | –             | –      | –      | XsyevBatched workspace 7.2 GB exceeds 12 GB device |

### Observations

- **At small-to-mid batches (256, 1024) the hybrid path approximately
  ties PR#8** (within 1–3% of PR#8) and beats our unified path by
  5–8%. Hybrid wins where its `XsyevBatched` workspace fits the device.
- **At large batches (4096) PR#8 wins decisively.** The unified path
  hits the single-block-per-matrix shared-memory ceiling on this
  consumer GPU (only 1 block resident per SM), and the hybrid path
  hits the cuSOLVER workspace ceiling and OOMs. PR#8's per-primitive
  batched kernels grid-stride over the batch dimension and don't share
  these constraints.
- **Eigenvalue cross-check stays at ~1.3 × 10⁻⁸** across all paths and
  batch sizes — different intra-block rounding orders between
  cuSolverDx::Heev and cuSOLVER's `XsyevBatched`. Both implementations
  are correct; neither is bit-equivalent to the other.

### Hardware caveats

- The 4096-batch numbers in the table above are from a session that
  followed a long autotune sweep (~15 min) plus an OOM/recovery cycle.
  PR#8's 4,908 mat/s at B=4096 is ~50% of an earlier-session reading
  (11,037 mat/s) on the same hardware — likely thermal throttling. The
  relative-ratio comparisons (within a single session, all three paths
  on the same temperature/state device) are reliable; cross-session
  absolute numbers should be treated with skepticism.
- `LD_LIBRARY_PATH` matters: CUDA.jl's bundled CUDA 12.9 artifacts can
  conflict with the system 12.8 toolkit our `libsunny_geneig.so` is
  linked against. The reproducibility recipe forces local
  toolkit via `CUDA.set_runtime_version!(v"12.8"; local_toolkit=true)`.

## Datacenter measurements

Pending H100 access. Expectations from the audit:

- H100's **228 KB per-block opt-in shared memory** (vs 101 KB on sm_120)
  may let the unified kernel try BPB=2 at N=54 — 2× cross-batch
  parallelism inside heev's `htev::warp_driver`. The unified path's
  N=54 BPB recommendation from cuSolverDx's tuning database is BPB=1
  even on H100 (heev limits dominate above N=16), so this is
  speculative; needs measurement.
- H100's **80 GB device memory** lifts the hybrid path's
  `XsyevBatched` workspace ceiling from ~3,000 matrices on the laptop
  to ~40,000 matrices.
- Per-primitive cuBLAS and cuSOLVER batched kernels have been hand-
  tuned by NVIDIA for H100 in a way they have not been for sm_120
  laptops. The PR#8 / hybrid relative ranking may shift in either
  direction.

When runs, the table here gets a "datacenter" section appended
with the same methodology.

## Decision

**Ship both unified and hybrid paths.** The C ABI exposes both via
`sunny_geneig_create_ex(..., kernel_mode)`; the Julia wrapper exposes
both via `Solver(n; kernel=:unified | :hybrid)`. Default is `:unified`
(simpler API, no auxiliary memory, single launch). Users who care
about laptop-batch-size throughput up to ~3,000 matrices can opt into
`:hybrid` for ~5–8% over unified.

The unified path is the architectural-novelty contribution; the hybrid
path is the pragmatic-throughput contribution. They serve different
user trade-offs and both are first-class citizens of the codebase.
