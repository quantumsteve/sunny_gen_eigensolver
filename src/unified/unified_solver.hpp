// runtime-compiled generalized eigenvalue solver.
//
// Constructor compiles the unified five-stage NVRTC kernel for the supplied
// matrix size N and a chosen BatchesPerBlock (BPB). BPB defaults to auto-
// select: try BPB ∈ {16, 8, 4, 2, 1} from largest to smallest, accept the
// first whose required shared memory fits the device's per-block dynamic
// shared-memory ceiling (with a 2 KB safety margin). Pass an explicit
// `batches_per_block > 0` to override.
//
// Caller contract: when launching, `batch_size` should be a multiple of
// `batches_per_block()` — the kernel does not bounds-check the partial last
// block.

#pragma once

#include <cuda.h>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>
#include <vector_types.h>  // dim3

// helpers — create/destroy a vector of CUstreams with default
// flags. These do not depend on solver state (free functions); the test
// harness uses them so it doesn't have to repeat the cuStreamCreate/Destroy
// loop.
std::vector<CUstream> make_streams(int n);
void                  destroy_streams(std::vector<CUstream>& streams);

// BlockDim selection strategy.
//   Suggested: use cuSolverDx's `suggested_block_dim` (probe-driven).
//   Autotune:  sweep candidate BlockDim values around `suggested`, measure
//              each on a 1024-matrix tuning workload (5-sample median),
//              validate output against the suggested-candidate reference,
//              pick the empirically fastest correctness-validated value.
//              Process-static cache memoizes the result per (N, BPB, arch).
enum class TuningMode { Suggested, Autotune };

// module cache strategy.
//   Auto:    in-process cache + on-disk cubin cache (production default).
//   NoDisk:  in-process cache only; never read or write the disk cache
//            (useful for testing the in-process cache in isolation).
//   NoCache: skip both caches; every construction does a full NVRTC compile
//            (only useful for first-compile correctness/timing tests).
enum class CacheMode { Auto, NoDisk, NoCache };

// Cached module entry. Lifetime is managed by std::shared_ptr; the map in
// `UnifiedGeneigSolver` holds std::weak_ptr so an entry is reclaimed when the
// last solver instance for that key is destroyed. The destructor unloads
// the CUmodule on the device.
struct CachedModule {
    CUmodule     module           = nullptr;
    CUfunction   kernel           = nullptr;
    dim3         block_dim        {0, 0, 0};
    unsigned int shared_mem_bytes = 0;
    int          batches_per_block= 0;
    int          n                = 0;
    int          block_dim_x      = 0;

    CachedModule() = default;
    CachedModule(const CachedModule&)            = delete;
    CachedModule& operator=(const CachedModule&) = delete;
    ~CachedModule();
};

// Cache key. Two solvers with the same (n, bpb, arch, block_dim_x) share
// the same compiled CUmodule. Adding fields here means "different value
// requires recompile."
struct ModuleCacheKey {
    int n             = 0;
    int bpb           = 0;
    int arch          = 0;
    int block_dim_x   = 0;
    bool operator<(const ModuleCacheKey& o) const {
        return std::tie(n, bpb, arch, block_dim_x)
             < std::tie(o.n, o.bpb, o.arch, o.block_dim_x);
    }
};

class UnifiedGeneigSolver {
public:
    // Construct solver for matrix size `n` on `device_id`. If
    // `batches_per_block_request <= 0`, auto-select the largest BPB that
    // fits the device's shared-memory ceiling. Otherwise honor the explicit
    // value (and throw if it doesn't fit).
    //
    // `force_block_dim_x > 0` is a testing-only escape hatch that skips both
    // the probe and any autotune sweep, compiling the production kernel
    // directly with that BlockDim.x. Use 0 (default) to let the probe (or
    // autotune) decide.
    //
    // `tuning_mode` selects how BlockDim.x is chosen when
    // `force_block_dim_x == 0`: Suggested (default, 3.5b path) reads
    // cuSolverDx's recommendation; Autotune (3.5e path) sweeps candidates.
    //
    // `cache_mode` (3.5f) enables/disables module caching. Default Auto
    // uses both the in-process map and an on-disk cubin cache.
    explicit UnifiedGeneigSolver(int n,
                               int device_id,
                               int batches_per_block_request = 0,
                               int force_block_dim_x = 0,
                               TuningMode tuning_mode = TuningMode::Suggested,
                               CacheMode  cache_mode  = CacheMode::Auto);
    ~UnifiedGeneigSolver();

    UnifiedGeneigSolver(const UnifiedGeneigSolver&)            = delete;
    UnifiedGeneigSolver& operator=(const UnifiedGeneigSolver&) = delete;
    UnifiedGeneigSolver(UnifiedGeneigSolver&&)                 = delete;
    UnifiedGeneigSolver& operator=(UnifiedGeneigSolver&&)      = delete;

    // Full generalised eigenvalue solve. `batch_size` is the total number
    // of matrices; the launcher computes gridDim = (batch_size + BPB - 1)/BPB.
    void launch(CUdeviceptr d_H, CUdeviceptr d_S,
                CUdeviceptr d_W, CUdeviceptr d_U, CUdeviceptr d_info,
                int batch_size, CUstream stream);

    // multi-stream chunked launcher. See nvrtc_solver.cpp.
    void launch_chunked(CUdeviceptr d_H, CUdeviceptr d_S,
                        CUdeviceptr d_W, CUdeviceptr d_U, CUdeviceptr d_info,
                        int total_matrices, int chunk_size, int num_streams,
                        const std::vector<CUstream>& streams);

    int          matrix_size()           const { return n_; }
    int          batches_per_block()     const { return batches_per_block_; }
    dim3         block_dim()             const { return block_dim_; }
    unsigned int shared_mem_bytes()      const { return shared_mem_bytes_; }

    // introspection.
    int          autotuned_block_dim_x() const { return autotuned_block_dim_x_; }
    bool         was_autotuned()         const { return was_autotuned_; }
    bool         autotune_sweep_ran()    const { return autotune_sweep_ran_; }

    // introspection. Returns the cache layer that satisfied
    // construction's production module: "in-process", "disk", or "compile"
    // (compile means cache miss followed by a full NVRTC + nvJitLink).
    const char*  cache_layer_used()      const { return cache_layer_used_; }
    // Wall time spent inside `acquire_module_` (in milliseconds), exposed
    // for the cache benchmark test.
    double       acquire_module_ms()     const { return acquire_module_ms_; }

    // return the resolved cache directory path. Used by tests
    // to clean state and assert disk artifacts. Reads SUNNY_GENEIG_CACHE_DIR
    // first; otherwise falls back to platform-default.
    static std::string cache_dir();
    // Compute the on-disk cubin path for a given key and the current
    // library/source content hashes. Used by tests for corruption injection.
    static std::string cubin_path(const ModuleCacheKey& key);

private:
    // Probe-compile a small kernel that exposes per-operator
    // `suggested_block_dim` via `__constant__`. Returns the maximum .x value
    // across the five operators. Throws on NVRTC compile failure.
    unsigned int probe_suggested_block_dim_x_(int bpb);

    // Build the production CUmodule for (n, bpb, block_dim_x) by NVRTC +
    // nvJitLink. Mutates the instance's module_/kernel_/block_dim_/
    // shared_mem_bytes_ fields directly. Used during the autotune sweep
    // where we deliberately do NOT route through the cache layer.
    void compile_production_(int bpb, unsigned int block_dim_x);

    // autotune sweep. See nvrtc_solver.cpp.
    unsigned int autotune_block_dim_x_(int bpb, unsigned int suggested);

    // produce the production module honoring the cache mode.
    // Sets module_handle_ (the lifetime-owning shared_ptr) and copies its
    // fields into the instance's launch-side state (kernel_, block_dim_,
    // shared_mem_bytes_). Records `cache_layer_used_` so the test harness
    // can verify the right path was taken.
    void acquire_module_(int bpb, unsigned int block_dim_x, CacheMode cache_mode);

    // Helpers used by acquire_module_.
    std::shared_ptr<CachedModule>
         build_module_from_cubin_(const std::vector<char>& cubin,
                                  int bpb, unsigned int block_dim_x);
    std::vector<char>
         compile_to_cubin_(int bpb, unsigned int block_dim_x);
    std::shared_ptr<CachedModule>
         try_load_disk_(const ModuleCacheKey& key);
    void write_disk_(const ModuleCacheKey& key,
                     const std::vector<char>& cubin);

    int          n_                      = 0;
    int          device_id_              = 0;
    int          batches_per_block_      = 0;
    int          device_max_smem_        = 0;
    CUcontext    context_                = nullptr;

    // 3.5f: lifetime-owning handle. Destruction (last shared_ptr ref drop)
    // unloads the CUmodule.
    std::shared_ptr<CachedModule> module_handle_;

    // Raw fields kept for fast launch path (no shared_ptr deref). These
    // alias module_handle_->{kernel, block_dim, shared_mem_bytes} after
    // acquire_module_ runs.
    CUmodule     module_                 = nullptr;
    CUfunction   kernel_                 = nullptr;
    dim3         block_dim_              {0, 0, 0};
    unsigned int shared_mem_bytes_       = 0;

    // 3.5e introspection.
    int          autotuned_block_dim_x_  = 0;
    bool         was_autotuned_          = false;
    bool         autotune_sweep_ran_     = false;

    // 3.5f introspection.
    const char*  cache_layer_used_       = "compile";
    double       acquire_module_ms_      = 0.0;

    // Process-static autotune-winner cache (3.5e).
    static std::map<std::tuple<int, int, int>, int> autotune_cache_;

    // Process-static module cache (3.5f). Holds weak_ptr so entries
    // disappear when their last owning solver is destroyed.
    static std::map<ModuleCacheKey, std::weak_ptr<CachedModule>> in_process_cache_;
    static std::mutex                                            in_process_cache_mutex_;
};
