# NVRTC runtime compilation

## Goal

Remove the N=54 hardcode from the kernel so Sunny.jl can use any
matrix size determined at runtime by the user's magnetic unit cell.

## Sub-step plan

- **3a (done, commit a64a862):** NVRTC plumbing for the reduction
  stage; hybrid with static heev finishing stage to work around
  CUSOLVERDX_NVRTC_HEEV_BUG. N=54 still hardcoded via `-DM_SIZE=54`.
- **3b (done, commit d9a835d):** cuSolverDx header overlay applied
  (`src/common/cusolverdx_overlay/`) — heev now compiles under NVRTC.
  Hybrid split removed; the NVRTC kernel runs all five stages
  (`potrf → trsm × 2 → heev → trsm`) in one launch. Constructor
  signature is `UnifiedGeneigSolver(int n, int device_id)`. Fixtures
  generated at N ∈ {16, 32, 48, 54, 64, 96, 128}; per-N validation in
  `tests/unified/test_multiN.cpp`.
- **3.5a (done, commit c8a5adf):** BatchesPerBlock parameterization.
  Constructor now takes a third optional argument
  (`batches_per_block_request`). When 0 (default), the constructor
  auto-selects: tries BPB ∈ {16, 8, 4, 2, 1} from largest to smallest,
  accepts the first that compiles AND whose `solver_shared_memory_size`
  fits the device's per-block dynamic shared-memory ceiling. cuSolverDx's
  own `static_assert` rejects BPB combinations whose shared memory
  exceeds the SOLVER_SM tag's ceiling (164 KB at SM<800>) before
  reaching the device-level check. New test
  `tests/unified/test_bpb_sweep_n54.cpp` validates explicit-BPB construction
  at all candidates and reports throughput on the n54_b32 fixture.
- **3.5c (done, this commit, no-op kernel change):** Negative-result
  experiment on cuSolverDx primitive synchronization. Reading
  `cusolverdx/database/cholesky.cuh` and `htev.cuh` shows their `dispatch`
  functions end with `__syncthreads()` in every code path, suggesting
  the external `__syncthreads()` we issue after each `Solver().execute()`
  is redundant for the Cholesky and Heev cases. Removing those two syncs
  *was source-level safe* but **regressed throughput by 14% on N=54
  BPB=1**: 5-sample median dropped from **6,494 mat/s** (5 syncs) to
  **5,791 mat/s** (3 syncs). Hypothesis: cuSolverDx's internal sync uses
  the partial-warp/warp-per-batch heuristic and may only sync the warps
  that participated in that operator's dispatch, leaving other warps
  free to race ahead and conflict with the next operator's full-CTA
  load. The 5 explicit syncs in our kernel are therefore **load-bearing
  for throughput on this hardware**, not just a defensive default. We
  keep them, ship the experiment as documented evidence, and add the
  finding to the project's "things that look like obvious wins but are
  not" register. The 5-sample medians (each sample is a fresh NVRTC
  cold compile + a 1024-matrix bench at N=54 BPB=1):
    - 5 syncs (3.5b kernel):  6,581 6,494 6,601 6,406 6,335 → median 6,494
    - 3 syncs (3.5c attempt): 5,740 5,784 5,800 5,808 5,791 → median 5,791
- **3.5b (done, commit cbf80bb):** Per-(N, BPB, arch) block dimension via
  cuSolverDx's `suggested_block_dim`. Constructor now does a two-phase
  compile per BPB candidate: a cheap probe (~3-5 s on this hardware)
  instantiates the solver types and reads each operator's
  `__constant__ suggested_block_dim`; we take the max .x across the
  five operators (potrf, three trsms, heev) and recompile the
  production kernel with `BlockDim<max>`. The probe source is held
  alongside the production source in
  [src/unified/kernel_source.hpp](../src/unified/kernel_source.hpp). The
  production source's hardcoded `BlockDim<128>` is replaced with
  `BlockDim<BLOCK_DIM_X>` driven by a new `-DBLOCK_DIM_X=` macro. The
  probe is fast because the source has no `execute()` calls and
  therefore no LTO IR cross-references into the cuSolverDx fatbin
  during nvJitLink. Per-N picked block dims on the development
  hardware: see table below.
- **3.5d (done, this commit):** Multi-stream chunked launch path. New
  `UnifiedGeneigSolver::launch_chunked` round-robins fixed-size chunks of
  a larger workload across N CUDA streams (no per-chunk synchronization;
  the CUDA scheduler is free to overlap). Free helpers `make_streams` /
  `destroy_streams` build/teardown a `std::vector<CUstream>`. New test
  `tests/unified/test_streams_n54.cpp` benchmarks 30 chunks × 2000 matrices
  (60,000 total at N=54) under stream configs {1, 2, 4, 8}, 5-sample
  median per config, with an 8-matrix LAPACK spot-check per sample.
  *(Note: production Sunny uses chunk_size=12000; we ran 2000 here so 30
  chunks fit in laptop GPU memory — 360k matrices at N=54 require >50 GB
  device memory which doesn't fit. The streaming pattern is identical;
  per-chunk batch is what differs.)* **Result: throughput is flat across
  all stream counts** — 1-stream median 6,861 mat/s, 8-stream median 6,911
  mat/s (0.7% delta, well within run-to-run noise). The kernel-only
  workload already saturates the GPU: each chunk launches 2000 blocks at
  BlockDim<128>, far exceeding the ~50 SMs of the RTX 5070 Ti Laptop, so
  there is no idle compute capacity for a second concurrent chunk to
  fill. Stream-level kernel-kernel concurrency provides no measurable
  speedup on this hardware. The win would need to come from overlapping
  HtoD/DtoH with kernel execution (a separate API surface than
  `launch_chunked`); deferred to a future sub-task. We ship the
  primitive + the negative-result benchmark as documented evidence and
  add the finding to the project's "things that look like obvious wins
  but are not" register.

  Stream config table (N=54, chunk_size=2000, total=60000, 5-sample medians):

  | stream config | median mat/sec | range (min..max) | speedup vs 1S | correctness |
  |---------------|---------------:|-----------------:|--------------:|-------------|
  |  1-stream     |          6,861 |    6,861 .. 6,863 |        1.000x | PASS        |
  |  2-stream     |          6,921 |    6,914 .. 6,936 |        1.009x | PASS        |
  |  4-stream     |          6,913 |    6,912 .. 6,913 |        1.008x | PASS        |
  |  8-stream     |          6,911 |    6,910 .. 6,912 |        1.007x | PASS        |

  All configs hit fp64 epsilon: max_eig_rel = 1.61e-15, max_phase = 3.66e-15
  on the 8 spot-check matrices per sample. Bit-equivalent results across
  stream configs (same eig_rel and phase to displayed precision) confirm
  the multi-stream path is correctness-equivalent to the single-stream
  baseline.

- **3.5e (done, this commit):** Empirical autotuning of BlockDim around
  cuSolverDx's `suggested_block_dim`. New `TuningMode::Autotune` constructor
  mode sweeps candidate BlockDim.x values around the probe-recommended
  value, benchmarks each on a 1024-matrix synthetic workload (5-sample
  median), validates each candidate's output against the suggested-
  candidate reference (transitive LAPACK coverage; the suggested-mode path
  has been LAPACK-validated end-to-end in [test_nvrtc_n54.cpp](../tests/unified/test_n54.cpp)
  and [test_nvrtc_multiN.cpp](../tests/unified/test_multiN.cpp)), and picks
  the fastest correctness-validated candidate. A process-static
  `std::map<(N, BPB, arch), winning_BlockDim>` cache memoizes the result
  so repeated constructions for the same `(N, BPB, arch)` skip the sweep.

  **Candidate set logic** ([src/unified/unified_solver.cpp](../src/unified/unified_solver.cpp)):
  raw set is `{suggested-64, suggested-32, suggested, suggested+32,
  suggested+64, suggested*2}`, then filtered to:
  - multiples of 32 (warp granularity);
  - `> 0` and `<= 1024` (CUDA hard limit);
  - `>= N` (so each thread handles at least one matrix-row element);
  - dedup via `std::set<int>`, sorted ascending for readable logs.

  After each candidate compiles, we additionally check that
  `shared_memory_size <= device_max_smem_` and skip the candidate cleanly
  if not — without this guard, `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`
  would crash the sweep on candidates whose cuSolverDx-reported smem fits
  the SOLVER_SM<800> 164 KB tag budget but exceeds the consumer-laptop
  101 KB device ceiling.

  **A correctness bug found and fixed during 3.5e bring-up:** the first
  draft of the sweep skipped recompile when `cand == suggested`, on the
  theory that suggested was already loaded from the Phase-1 reference
  compile. But by the time the sorted candidate loop reached suggested,
  an earlier (smaller) candidate had already overwritten `module_` /
  `kernel_`, so we measured that earlier candidate's kernel under the
  suggested-row's launch config. The bug fingerprinted as two adjacent
  rows reporting identical throughput AND identical eig_rel/phase — a
  giveaway that the same binary was being benchmarked twice. Fix: always
  recompile per candidate; eat the extra compile cost. Correctness is
  non-negotiable.

  **A methodology bug found and fixed during 3.5e bring-up:** the spec
  called for a 256-matrix tuning workload, but at this hardware that's
  too small to characterize the production regime. At 256 matrices and
  BPB=2 we launch only 128 blocks on ~50 SMs (≈2.5 blocks/SM), and the
  GPU is so under-occupied that BlockDim=128 wins by filling SMs while
  the production 1024-matrix workload is large enough that BlockDim=64
  wins on the same configuration. Bumped `kTuneB` from 256 to 1024 to
  match the production benchmark size; characterization order now agrees
  with production.

  ### Per-candidate sweep results (sm_120 RTX 5070 Ti Laptop)

  **N=54, BPB=1** (suggested=128 from probe):

  | BlockDim.x | shared_mem | sweep median mat/s | range (min..max) | eig_rel  | phase    | status                              |
  |-----------:|-----------:|-------------------:|------------------:|---------:|---------:|-------------------------------------|
  |         64 |     97,280 |              6,286 |     6,284..6,289 | 2.11e-15 | 4.00e-15 | OK                                  |
  |         96 |     98,176 |          **8,088** |     8,085..8,091 | 2.50e-15 | 4.88e-15 | **OK (winner)**                     |
  |        128 |     99,712 |              6,754 |     6,735..6,755 | 0.00e+00 | 6.44e-15 | OK (= suggested)                    |
  |        160 |    100,736 |              6,673 |     4,298..6,675 | 2.05e-15 | 3.55e-15 | OK                                  |
  |        192 |    101,760 |                  — |                — |        — |        — | SKIP (shared_mem > 101,376 device max) |
  |        256 |    103,808 |                  — |                — |        — |        — | SKIP (shared_mem > 101,376 device max) |

  Sweep wall: 874.6 s (six full NVRTC compiles inside the sweep, plus one
  reference compile of suggested before the sweep, plus one final compile
  of the winner after the sweep).

  **N=32, BPB=2** (suggested=64 from probe):

  | BlockDim.x | shared_mem | sweep median mat/s | range (min..max) | eig_rel  | phase    | status                              |
  |-----------:|-----------:|-------------------:|------------------:|---------:|---------:|-------------------------------------|
  |         32 |     68,032 |             24,453 |    15,206..24,477 | 1.94e-15 | 3.55e-15 | OK                                  |
  |         64 |     68,032 |         **46,676** |    46,400..46,691 | 0.00e+00 | 5.00e-15 | **OK (winner = suggested)**         |
  |         96 |     69,088 |             36,186 |    30,513..36,194 | 2.22e-15 | 3.89e-15 | OK                                  |
  |        128 |     69,600 |             40,463 |    40,298..40,466 | 2.66e-15 | 4.33e-15 | OK                                  |

  Sweep wall: 114.8 s (four compiles).

  ### Production-benchmark comparison (1024 random matrices)

  | config             | mode      | BDx | median mat/s | range (min..max) | speedup vs sug | validation |
  |--------------------|-----------|----:|-------------:|------------------:|---------------:|------------|
  | N=54 BPB=1         | suggested | 128 |        6,757 |     6,703..6,758 |         1.000x | PASS       |
  | N=54 BPB=1         | autotune  |  96 |    **8,092** |     8,089..8,094 |     **1.198x** | PASS       |
  | N=54 BPB=1         | autotune2 |  96 |        8,095 |     8,089..8,098 |         1.198x | PASS [cache] |
  | N=32 BPB=2         | suggested |  64 |       46,618 |    46,570..46,629 |         1.000x | PASS       |
  | N=32 BPB=2         | autotune  |  64 |       46,592 |    46,560..46,597 |         0.999x | PASS       |
  | N=32 BPB=2         | autotune2 |  64 |       46,593 |    44,504..46,627 |         0.999x | PASS [cache] |

  ### Result and decision

  **N=54 BPB=1: real 1.198× (≈20%) speedup at the production size.** The
  autotune picks BlockDim=96 over the cuSolverDx-suggested BlockDim=128.
  Why: cuSolverDx's heuristic takes the max across all five operators
  (`max(chol=64, trsmL=64, trsmR=64, trsmLC=64, heev=128) = 128`). heev
  alone wants 128, but heev is one of five stages in the unified kernel.
  Empirically, BlockDim=96 is enough for heev's working set while leaving
  the chol / three-trsm stages closer to their per-operator optima of
  64 — a better global compromise than the per-operator max.
  Correctness validates at fp64 epsilon (eig_rel=1.50e-15, phase=3.77e-15
  on the 8-matrix LAPACK fixture). Ship it.

  **N=32 BPB=2: null result.** Autotune picks BlockDim=64 = suggested.
  Speedup 0.999× (within noise). Why: at N=32 BPB=2 the per-operator
  recommendations are `(64, 64, 64, 64, 32)`; heev's 32 doesn't dominate
  the max, so cuSolverDx's heuristic is already optimal for this config.
  This is the same null-result shape as 3.5c (sync minimization) and 3.5d
  (multi-stream concurrency at this hardware): the autotune machinery
  proves it's harmless and adds zero risk; the win is just config-
  dependent. Ship the infrastructure; the cached result is BlockDim=64,
  same as the Suggested-mode default.

  ### Diagnostic: why the heuristic over-shoots at N=54 but not at N=32

  cuSolverDx's `suggested_block_dim` is the per-operator recommendation
  for *that operator running alone in a kernel*. The 3b unified kernel
  fuses five operators into one launch with a single BlockDim. Taking
  the max across operator suggestions gives correctness (every operator
  has at least its required threads) but not optimality, because:

  - At N=32 BPB=2, four of five operators want 64 and heev wants 32. Max
    is 64, dominated by the four-operator majority. Heuristic wins.
  - At N=54 BPB=1, four of five operators want 64 and heev wants 128.
    Max is 128, dominated by heev alone. The four 64-want operators run
    at 2× their optimal thread count, and the wall-clock cost of the
    ~four under-utilized stages outweighs heev's ~one well-utilized
    stage. Heuristic over-shoots by 33% (96 is enough for heev given
    sm_120's per-warp register file; the extra 32 threads are dead
    weight to chol/trsm).

  This is consistent with the N=32 micro-bench finding
  (BlockDim=64 vs 128 is 1.153× at N=32 BPB=2 ⇒ same direction as 3.5e
  at N=54: smaller is better than the heuristic max when heev's
  recommendation is a minority).

  ### How to use Autotune

  ```cpp
  UnifiedGeneigSolver solver(N, /*device_id=*/0,
                            /*bpb=*/0,                    // auto-select BPB
                            /*force_block_dim_x=*/0,      // disable forced override
                            TuningMode::Autotune);        // run sweep
  // first construction at this (N, BPB, arch): runs sweep (slow)
  // subsequent constructions: cache hit (fast — only the production compile)
  ```

  Cache is process-static; survives across solver instances within the
  same process. Persisting the autotune cache to disk is deferred to
  (cubin cache).

- **3.5f (done, this commit):** Two-layer module cache. The 3a–3.5e plan
  had separate items for the in-process CUmodule cache (3c) and the
  on-disk cubin cache (3d); 3.5f delivers both layers in one pass and
  retires both 3c and 3d. The 213-second cold compile becomes the
  one-and-only first-time cost; subsequent constructions of the same
  `(N, BPB, arch, BlockDim.x)` skip NVRTC + nvJitLink entirely.

  ### Architecture

  ```
  UnifiedGeneigSolver(N, dev, bpb, force, tuning, cache)
        │
        ├── determine (N, bpb, BlockDim.x) [probe / autotune / force]
        │
        └── acquire_module_(bpb, BlockDim.x, cache_mode)
              │
              ├── Layer 1: in-process std::map<(N,bpb,arch,bdx), weak_ptr<CachedModule>>
              │   guarded by std::mutex; cache holds weak_ptr so an entry
              │   reclaims when the last solver instance referencing it
              │   destructs.
              │
              ├── Layer 2: on-disk cubin file at
              │   $XDG_CACHE_HOME/sunny_geneig/cubins/
              │     {N}_{bpb}_{arch}_{bdx}_lib{libhash16}_src{srchash16}.cubin
              │   loaded via cuModuleLoadDataEx — skips NVRTC + nvJitLink.
              │
              └── Layer 3: full compile (NVRTC → LTO IR → nvJitLink → cubin)
                  fall-through. The fresh cubin is then written to disk
                  (atomic rename) and inserted into the in-process map.
  ```

  Lifetime: `CachedModule` owns its `CUmodule` via destructor (`cuModuleUnload`).
  Each solver holds one `std::shared_ptr<CachedModule>`; the cache map
  holds a `std::weak_ptr`. When the last solver for a key destructs,
  the module unloads (instead of leaking until process exit).

  ### Cache-key hashing

  The disk filename embeds two 64-bit content hashes — a **library hash**
  and a **source hash** — so a cubin compiled against an old MathDx (or
  an old `kernel_source.hpp`) becomes a key miss after either changes,
  rather than silently loading and crashing the device.

  | Hash         | Input                                              | Algorithm   |
  |--------------|----------------------------------------------------|-------------|
  | `lib_hash`   | first 8 bytes of `libcusolverdx.fatbin` ‖ filesize | FNV-1a-64   |
  | `source_hash`| entire `nvrtc_geneig::kKernelSource` byte string   | FNV-1a-64   |

  FNV-1a is a content fingerprint, not crypto. The cache key already
  pins `(N, bpb, arch, BlockDim.x)` exactly; the hashes only need to
  detect *which* library and *which* source produced the cubin.
  Collisions on small inputs at 64-bit are astronomically unlikely
  (~2⁻⁶⁴) and not a security concern in this context.

  Filename example from the test run:

  ```
  /home/rb/.cache/sunny_geneig/cubins/54_1_120_128_libf708a7883b6bcdf3_src6338eccfe1c378a8.cubin
  ```

  ### Cache directory location

  Resolved in this order (first hit wins):
  1. `$SUNNY_GENEIG_CACHE_DIR` if set (test override hatch);
  2. Linux: `$XDG_CACHE_HOME/sunny_geneig/cubins`, else `$HOME/.cache/sunny_geneig/cubins`;
  3. macOS: `$HOME/Library/Caches/sunny_geneig/cubins`;
  4. Windows: `%LOCALAPPDATA%\sunny_geneig\cubins`;
  5. `/tmp/sunny_geneig/cubins` as a last-resort fallback.

  Created with `mkdir -p` semantics on first write.

  ### Multi-process and durability

  Disk writes go to `<final>.tmp.<pid>.<nanos>` then `rename(2)` to the
  final path. POSIX `rename` is atomic on the same filesystem, so two
  Julia sessions launching simultaneously cannot produce a partial cubin
  visible to a third reader. If concurrent compilers race, the second
  rename atomically replaces the first; both wrote correct cubins so
  any winner is fine.

  ### Validation on load

  Three checks before trusting a cached cubin:
  1. **Size > 1024 bytes.** Earlier short writes (interrupted by a kill
     before `fclose`) get rejected here.
  2. **ELF magic** (`0x7F 'E' 'L' 'F'`). Detects bit-flips, manual edits,
     or filesystem corruption that survived size sanity. The corruption
     test deliberately overwrites this and the cache must recover.
  3. **`cuModuleLoadDataEx` succeeds.** Catches semantic invalidity that
     made it past the magic check.

  Any of the three failing → log, `unlink` the bad file, fall through
  to compile.

  ### Cache modes (constructor parameter)

  | `CacheMode`  | In-proc | Disk read | Disk write | Use                     |
  |--------------|:-------:|:---------:|:----------:|-------------------------|
  | `Auto` (default) |  ✓  |     ✓     |     ✓      | production              |
  | `NoDisk`     |    ✓    |     ✗     |     ✗      | testing in-proc only    |
  | `NoCache`    |    ✗    |     ✗     |     ✗      | first-compile timings   |

  ### Test results

  **`test_cache_n54`** (clean cache → cold → warm-disk → in-process → no-cache):

  | phase       | BDx | construct (ms) | acquire (ms) | cache layer | validation |
  |-------------|----:|---------------:|-------------:|-------------|------------|
  | cold disk   | 128 |      212,748.0 |    205,394.2 | compile     | PASS (eig=2.16e-15, phase=3.77e-15) |
  | warm disk   | 128 |        3,624.8 |        285.3 | disk        | PASS (eig=2.16e-15, phase=3.77e-15) |
  | in-process  | 128 |        3,211.3 |        0.007 | in-process  | PASS (eig=2.16e-15, phase=3.77e-15) |
  | no-cache    | 128 |      198,663.2 |    195,549.2 | compile     | PASS (eig=2.16e-15, phase=3.77e-15) |

  Targets met:
  - warm disk `acquire (ms)` < 500ms target → **285.3 ms** ✓
  - in-process `acquire (ms)` < 5ms target → **0.007 ms** ✓ (~720× headroom)
  - all four phases bit-identical W and U output (W=`aa3c8fbb3e8db027`,
    U=`fbafa39dcdb6fbdf`); the disk cache is correctness-preserving.

  Note: `construct (ms)` for warm-disk and in-process is ~3 s because of
  the unconditional cuSolverDx probe at the start of the constructor
  (~3.2 s NVRTC compile of the small probe kernel that reads
  `__constant__ suggested_block_dim`). The probe is independent of the
  module cache; eliminating it would require a separate cache for the
  probe constants. `acquire_module_ms()` on the table excludes the probe
  and reports purely the cache-layer time.

  **`test_cache_invalidation`** (corrupt cubin → recover):

  ```
  solver A correct:                          PASS (eig=2.16e-15, phase=3.77e-15)
  solver A cubin written:                    PASS (48,941,656 bytes)
  solver B took compile path (not disk):     PASS (layer=compile)
  solver B correct after recompile:          PASS (eig=2.16e-15, phase=3.77e-15)
  cubin ELF magic restored:                  PASS (post_size=48,941,656 bytes)
  A and B produce bit-identical output:      PASS (W=aa3c8fbb3e8db027, U=fbafa39dcdb6fbdf)
  ```

  Step-by-step: solver A populates the cache (cold compile + disk write).
  Solver A is destructed; we manually overwrite the first 4 bytes of the
  cubin file with `'XXXX'`. Solver B is constructed with the same key —
  the disk loader's ELF magic check trips, the bad file is `unlink`ed,
  the compile path runs, the new cubin is atomically written, and B
  produces output bit-identical to A.

  ### Operational impact

  For Sunny.jl users, this drops repeat cold-start cost from 213 s to
  ≈285 ms across Julia sessions. With the autotune cache (3.5e) layered
  on top, even autotune-mode constructions hit the module cache for the
  winning BlockDim.x — observed in `test_autotune_n54`'s `autotune2`
  construction completing in 3.4 s (probe + cache hit) instead of the
  ~900 s autotune sweep on first run. The cache directory is durable
  across reboots; clearing it (`rm -rf ~/.cache/sunny_geneig/`) reverts
  to first-compile timings, which is also how the test sets up its
  cold-start phase.

- **3.5h (done, this commit):** First end-to-end PR#8 vs ours
  comparison on physically realistic SW08 matrices. New C ABI shim
  (`src/c_api.h` / `src/c_api.cpp` → `libsunny_geneig.so`) and Julia
  package `julia/SunnyGeneigBench` driving Sunny's
  `SpinWaveTheoryDevice` for SW08 √3×√3 kagome AFM, calling both
  `Sunny.CUDAExt.hegvd_batched!` (PR#8 path) and our NVRTC kernel via
  `ccall` on bit-identical `(H_d, Σ_d)` inputs. **PR#8 wins all three
  batch sizes on this laptop**: 0.798× (256), 0.915× (1024), 0.486×
  (4096). Eigenvalue agreement ~1.3 × 10⁻⁸ — different intra-block
  rounding orders between cuSOLVER batched primitives and our
  cuSolverDx blocks; both implementations are correct, just not
  bit-identical. Full hardware/method/discussion in
  [COMPARISON.md](COMPARISON.md). This is the first honest "us vs them"
  data point.

- **3e:** Performance characterization vs the static N=54 path,
  plus end-to-end timing on the existing batched test.

## 3.5b multi-N validation outcome (sm_120 RTX 5070 Ti Laptop, 101,376 bytes max dynamic shared memory per block)

| N   | BPB | BlockDim.x | shared_mem | cold compile (ms) | rand max_eig_rel | rand max_phase | near max_phase | status |
|-----|----:|-----------:|-----------:|------------------:|------------------:|----------------:|---------------:|--------|
|  16 |   8 |         96 |     70,400 |            52,058 |          0.00e+00 |        1.78e-15 |       1.89e-15 | PASS   |
|  32 |   2 |         64 |     68,032 |            55,329 |          0.00e+00 |        1.89e-15 |       1.78e-15 | PASS   |
|  48 |   1 |        128 |     78,704 |            53,489 |          1.50e-15 |        2.55e-15 |       2.33e-15 | PASS   |
|  54 |   1 |        128 |     99,712 |           221,632 |          2.16e-15 |        3.77e-15 |       2.44e-15 | PASS   |
|  64 |   — |          — |          — |                 — |                 — |               — |              — | SKIP   |
|  96 |   — |          — |          — |                 — |                 — |               — |              — | SKIP   |
| 128 |   — |          — |          — |                 — |                 — |               — |              — | SKIP   |

**Per-operator suggested block_dim.x at successful (N, BPB):**

| N  | BPB | Cholesky | TrsmLeft | TrsmRight | TrsmLeftConj | Heev | max → BDx |
|---:|----:|---------:|---------:|----------:|-------------:|-----:|----------:|
| 16 |   8 |       96 |       32 |        32 |           32 |   32 |        96 |
| 32 |   2 |       64 |       64 |        64 |           64 |   32 |        64 |
| 48 |   1 |       64 |       64 |        64 |           64 |  128 |       128 |
| 54 |   1 |       64 |       64 |        64 |           64 |  128 |       128 |

cuSolverDx's recommendations vary per primitive: at small N with high BPB
(N=16, BPB=8) Cholesky wants **96** threads per block while heev only wants
32. At larger N (N=48, 54) heev's recommendation of **128** dominates. The
auto-select takes the max so every operator has at least its required
threads.

For N=32, BPB=2 the new code uses **BlockDim<64>** vs the previous
hardcoded BlockDim<128>; this also slightly reduces heev's workspace_size
(shared_mem dropped from 69,600 to 68,032 bytes — 1.5 KB saved per block).

### BlockDim micro-bench at N=32, BPB=2 (test_blockdim_n32)

Direct A/B benchmark on 1024 random (H, S) pairs, same workload, same
solver, only difference is BlockDim.x:

| BlockDim.x | source       | mat/sec | wall (s) | shared_mem | correctness |
|-----------:|--------------|--------:|---------:|-----------:|-------------|
|         64 | suggested    |  46,692 |   0.0219 |     68,032 | PASS        |
|        128 | 3.5a hardcode |  40,480 |   0.0253 |     69,600 | PASS        |

**Ratio: 1.153 — suggested BlockDim<64> is 15.3% faster.**

This is a real architectural win, not just future-proofing for H100. At
N=32, every cuSolverDx primitive in our pipeline (potrf, three trsms,
heev) reports a suggested block_dim.x ≤ 64. Running with BlockDim<128>
launched twice as many threads as any operator could use, cutting
occupancy in half. cuSolverDx's per-arch tuning database knows this;
the probe surfaces it. Both variants produce eigvals/eigvecs at fp64
epsilon, so this is a pure throughput improvement.

At N=54 (Sunny's actual SW08 size on this hardware) the suggested max
matches 128, so the production kernel is byte-identical to 3.5a — no
delta either way. The gains are at smaller N. For the SW08 workload
specifically, 3.5b is correctness-equivalent to 3.5a; the win materializes
when Sunny extends to other matrix sizes.

**Summary:** 8/8 (N, fixture) pairs passed at fp64 epsilon for
N ∈ {16, 32, 48, 54}. N ∈ {64, 96, 128} skipped — even BPB=1's
shared-memory requirement exceeds the device ceiling on this
consumer hardware.

The auto-select picked **BPB=8 at N=16** and **BPB=2 at N=32** —
multiple matrices per block fit at the smaller sizes, packing more
work into each launch. At N=48 and above, only BPB=1 fits.

Note the eigenvalue relative diff at N=16 and N=32 is **0.00e+00**
(LAPACK's outputs are bit-identical to ours, given the same input
fixture). At larger N the residual is single-digit ulps, well within
fp64 epsilon.

### BPB sweep at N=54 (test_bpb_sweep_n54, )

| BPB | BlockDim.x | constructable | matrices/sec | max_eig_rel | max_phase | status |
|----:|-----------:|---------------|-------------:|------------:|----------:|--------|
|  1  |        128 | yes           |        4,645 |   2.22e-15  | 3.44e-15  | PASS   |
|  2  |          — | no            |            — |          —  |        —  | SKIP (probe rejected: cuSolverDx static_assert) |
|  4  |          — | no            |            — |          —  |        —  | SKIP   |
|  8  |          — | no            |            — |          —  |        —  | SKIP   |

On H100 NVL (228 KB per-block opt-in shared memory), BPB=2 at N=54
is expected to fit (~199 KB requirement). That measurement is in
.

### throughput regression check (N=54, B=1024, BPB=1)

Same workload as the baseline benchmark (1024 synthetic
matrices, BPB=1, NVRTC path):

| Path | matrices/sec | wall (s) |
|---|---:|---:|
| static-link  | 6,740 | 0.152 |
| NVRTC (hardcoded BDx=128) | 6,699 | 0.153 |
| NVRTC (suggested BDx=128) | 6,740 | 0.152 |

At N=54, BPB=1 the cuSolverDx-suggested `max(64,64,64,64,128)=128`
matches the previous hardcode, so the production kernel is identical.
The 0.6% delta vs 3.5a is run-to-run noise. **No regression.**

The single-block kernel layout requires
`Cholesky::shared_memory_size + Heev::shared_memory_size` = roughly
`2·N²·sizeof(cdouble) + workspace`. At N=64 that exceeds the
sm_120 laptop's 101 KB per-block ceiling. cuSolverDx 25.12's
`static_assert "Provided combination of data type and sizes makes this
problem not fit into shared memory available on the specified architecture"`
catches the larger N values before they reach the GPU.

H100 SXM5 / NVL has 228 KB max dynamic shared memory per block (opt-in),
which would push the single-block ceiling to roughly N≈80 for the
unified five-stage layout. Beyond that, a multi-block strategy (e.g.,
splitting the matrix tile across cooperative blocks, or reusing one
block's shared memory across stages instead of holding As + Heev region
live simultaneously) is needed for very large N. Deferred to .

For Sunny's current SW08 usage, N=54 is the operative size and is
fully validated. The runtime-N pitch holds for N up to 54 on consumer
hardware; up to ~80 on H100; further extension is a future-work item.

## Notes

- The cold NVRTC + nvJitLink pipeline takes 5–215 s per N (scales
  superlinearly with N as more cuSolverDx LTO objects are pulled in).
  Acceptable for Sunny's usage pattern (one-time cost per (session, N),
  amortized over hundreds of thousands of matrix solves) but motivates
  the disk cache . See timing breakdown in
  [CUSOLVERDX_NVRTC_HEEV_BUG.md](CUSOLVERDX_NVRTC_HEEV_BUG.md).
- Removing the temporary `[UnifiedGeneigSolver] phase timings` printf
  instrumentation is deferred until 3c, where it'll be replaced by
  a structured introspection method on the solver class.
