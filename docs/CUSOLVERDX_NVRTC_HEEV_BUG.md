# cuSolverDx 25.12 + NVRTC: `function::heev` instantiation fails

## Summary

When the `block_execution<...>` specialization for `Function<function::heev>` is
instantiated under NVRTC (CUDA 12.8), the static-constexpr member
`workspace_size`'s initializer references an unqualified identifier
`max_threads_per_block` that is declared *later* in the same class body.
NVRTC reports `max_threads_per_block` as undefined; downstream
`heev::suggested_batches` and `heev::suggested_block_dim` template
substitutions then fail with `<expression>` markers because the class is in
an error state.

The same code compiles cleanly under nvcc, where complete-class-context
name lookup for member function bodies makes order within the class body
irrelevant. NVRTC's stricter lookup is exposed by this declaration order.

`gesvd` is affected by the same code path; `potrf`, `trsm`, all the QR/LU
variants, etc. return 0 from `get_workspace_size()` and therefore never
trigger the lookup.

## Verified raw header text

`include/cusolverdx/detail/solver_execution.hpp` lines 197-205 — the
`get_workspace_size()` implementation:

```cpp
__host__ __device__ __forceinline__ static constexpr int get_workspace_size() {
    if constexpr (function_of_v<this_type> == function::heev) {
        return heev::workspace_size<a_cuda_data_type, base_type::m_size, job, max_threads_per_block, batches_per_block, base_type::this_sm_v>();
    } else if constexpr (function_of_v<this_type> == function::gesvd) {
        return gesvd::workspace_size<a_cuda_data_type, base_type::m_size, base_type::n_size, max_threads_per_block, batches_per_block, base_type::this_sm_v>();
    } else {
        return 0;
    }
}
```

Lines 685-694 — the static-constexpr declaration block of
`block_execution<Operators...>`:

```cpp
static constexpr unsigned int suggested_batches_per_block = get_suggested_batches_per_block();

static constexpr dim3 suggested_block_dim = get_suggested_block_dim();
static constexpr dim3 block_dim           = get_block_dim();

static constexpr unsigned int workspace_size = get_workspace_size();
static constexpr unsigned int shared_memory_size = get_shared_memory_size();

static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
static constexpr unsigned int min_blocks_per_multiprocessor = 1;
```

`workspace_size` (line 690) is initialized via `get_workspace_size()`, which
references `max_threads_per_block` (declared three lines below at line 693).

## Minimal repros (in this repo)

[`tests/repro_heev_nvrtc.cpp`](../tests/repro_heev_nvrtc.cpp) and
[`tests/repro_potrf_nvrtc.cpp`](../tests/repro_potrf_nvrtc.cpp) are
deliberately identical except for the `Function<>` operator. Both run
NVRTC with the standard cuSolverDx option set
(`--std=c++17 --device-as-default-execution-space -dlto
--relocatable-device-code=true --gpu-architecture=sm_NN` plus `--include-path`
flags for cuSolverDx, CUTLASS, and the CUDA toolkit). The kernel sources
do nothing but instantiate the `Solver` type and read
`Solver::shared_memory_size` to force class instantiation; no
`execute()` is called.

| Repro | Function operator | NVRTC result | Exit |
|---|---|---|---|
| `repro_heev_nvrtc`  | `Function<function::heev>()`  | **FAILED** with 6 errors (3 distinct, each emitted twice) | 1 |
| `repro_potrf_nvrtc` | `Function<function::potrf>()` | **SUCCEEDED**                                              | 0 |

The NVRTC log from the heev repro reports the three error sites with the
identical pattern observed in the project's main NVRTC kernel:
`solver_execution.hpp:143` (`heev::suggested_batches` no instance match),
`:177` (`heev::suggested_block_dim` no instance match), and `:199`
(`identifier "max_threads_per_block" is undefined`). The `:199` error
is the proximate cause; the other two are downstream cascades.

## Suggested upstream fix

Reorder the static-constexpr declarations of `block_execution` so
`max_threads_per_block` precedes `workspace_size`:

```diff
             static constexpr dim3 block_dim           = get_block_dim();

-            static constexpr unsigned int workspace_size = get_workspace_size();
-            static constexpr unsigned int shared_memory_size = get_shared_memory_size();
-
             static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
+
+            static constexpr unsigned int workspace_size = get_workspace_size();
+            static constexpr unsigned int shared_memory_size = get_shared_memory_size();
+
             static constexpr unsigned int min_blocks_per_multiprocessor = 1;
```

This is a mechanical reorder, no semantic change. Under nvcc behaviour is
unchanged; under NVRTC the heev/gesvd `get_workspace_size()` initializer
can resolve `max_threads_per_block` because it is now declared lexically
earlier.

## Project workaround status

**: applied via local header overlay.** The hybrid two-kernel split
that 3a used has been removed. The unified NVRTC kernel now runs all five
stages (`potrf → trsm × 2 → heev → trsm`) inside one launch.

The overlay lives at
[`src/common/cusolverdx_overlay/cusolverdx/detail/solver_execution.hpp`](../src/common/cusolverdx_overlay/cusolverdx/detail/solver_execution.hpp).
It is a verbatim copy of the upstream cuSolverDx 25.12 header with two
mechanical reorders inside `block_execution<Operators...>`:

1. `max_threads_per_block` declared before `workspace_size` (the original
   bug above).
2. The alias block (`type`, `a_arrangement`, `b_arrangement`, `transpose`,
   `fill_mode`, `side`, `diag`, `job`, `sm`) moved up next to
   `batches_per_block`. These were originally at the bottom of the class
   and are referenced by `get_suggested_batches_per_block()`,
   `get_suggested_block_dim()`, and `get_workspace_size()` member functions
   declared earlier in the class. NVRTC's name lookup does not see them
   when class-scope member-function bodies reference them out of declaration
   order; same root cause as the `max_threads_per_block` issue, surfaced
   only after the first reorder lifted the `max_threads_per_block` block.

Both reorders are mechanical, no semantic change. Verified end-to-end via
[`tests/repro_heev_nvrtc.cpp`](../tests/repro_heev_nvrtc.cpp) (fails
without overlay, succeeds with it; same source modulo include path).

The overlay is injected into NVRTC's include search path *before* the
upstream cuSolverDx include path, so `#include "cusolverdx/detail/solver_execution.hpp"`
resolves to our patched copy. Other cusolverdx headers come from upstream
unmodified.

**Removal procedure** when an upstream cuSolverDx release fixes both
declaration orders:
1. Delete `src/common/cusolverdx_overlay/`.
2. Remove `CUSOLVERDX_OVERLAY_INCLUDE_DIR` from
   `target_compile_definitions(nvrtc_geneig ...)` and the two repro
   targets in `tests/CMakeLists.txt`.
3. Remove the `overlay_inc` `--include-path` injection in
   [`src/unified/unified_solver.cpp`](../src/unified/unified_solver.cpp) and
   [`tests/repro_heev_nvrtc.cpp`](../tests/repro_heev_nvrtc.cpp).

The overlay applies to commit a64a862's hybrid split as well — should we
ever need to revert to the hybrid, the overlay is independent of which
kernel layout we ship.

## Related observation: nvJitLink performance

Independently of the heev bug, we measured nvJitLink wall time during
the construction of `UnifiedGeneigSolver`. For a 5 MB LTO IR (potrf +
2× trsm at N=54, complex<double>) linked against the cuSolverDx 25.12
fatbin on an RTX 5070 Ti Laptop GPU (sm_120, CUDA 12.8):

| Phase           | Cold (ms) | Warm (ms) |
|-----------------|-----------|-----------|
| ctx-init        | 2,858     | 0         |
| nvrtcCompile    | 2,817     | 1,380     |
| GetLTOIR        | 1         | 0         |
| nvJitLink       | 79,452    | 72,521    |
| cubin load      | 213       | 203       |
| TOTAL           | 85,343    | 74,103    |

nvJitLink consumes 93-98% of total compile time. Output cubin is 27 MB.
This may indicate that nvJitLink is performing LTO across the entire
cuSolverDx fatbin even when only three primitives are referenced from
the LTO IR. NVIDIA's `nvrtc_potrs.cpp` example (which uses one primitive)
reportedly compiles in seconds.

This is a separate observation from the heev declaration-order bug;
it does not block correctness. Mitigations like cubin-on-disk caching
are deferred to a later phase.
