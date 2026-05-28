// C ABI implementation. Wraps UnifiedGeneigSolver in an
// opaque struct, converts C++ exceptions to error codes, and exposes
// device pointers as uintptr_t for FFI portability.

#include "common/c_api.h"
#include "unified/unified_solver.hpp"
#include "hybrid/hybrid_solver.hpp"

#include <cuda.h>

#include <exception>
#include <memory>
#include <new>
#include <string>

// solver handle now wraps either UnifiedGeneigSolver
// (unified) or HybridGeneigSolver. The C ABI doesn't expose this
// distinction beyond the create-time kernel_mode selector.
struct sunny_geneig_solver {
    sunny_geneig_kernel_mode_t mode = SUNNY_GENEIG_KERNEL_UNIFIED;
    std::unique_ptr<UnifiedGeneigSolver>  unified;
    std::unique_ptr<HybridGeneigSolver> hybrid;

    int n_for_intro            = 0;
    int bpb_for_intro          = 0;
    int bdx_for_intro          = 0;
    unsigned int smem_for_intro = 0;

    sunny_geneig_solver(int n, int device_id, int bpb_request,
                        TuningMode tuning, CacheMode cache)
        : mode(SUNNY_GENEIG_KERNEL_UNIFIED) {
        unified = std::make_unique<UnifiedGeneigSolver>(
            n, device_id, bpb_request,
            /*force_block_dim_x=*/0, tuning, cache);
        n_for_intro    = unified->matrix_size();
        bpb_for_intro  = unified->batches_per_block();
        bdx_for_intro  = (int)unified->block_dim().x;
        smem_for_intro = unified->shared_mem_bytes();
    }

    sunny_geneig_solver(int n, int device_id,
                        sunny_geneig_kernel_mode_t /*hybrid_marker*/)
        : mode(SUNNY_GENEIG_KERNEL_HYBRID) {
        hybrid = std::make_unique<HybridGeneigSolver>(n, device_id);
        n_for_intro    = hybrid->matrix_size();
        bpb_for_intro  = 1;
        bdx_for_intro  = hybrid->block_dim_x();
        smem_for_intro = hybrid->kernel_shared_mem_bytes();
    }
};

namespace {

thread_local std::string g_last_error;

void set_last_error(const char* msg) {
    g_last_error.assign(msg ? msg : "");
}

void clear_last_error() {
    g_last_error.clear();
}

CUstream stream_from_uintptr(uintptr_t s) {
    return reinterpret_cast<CUstream>(s);
}

CUdeviceptr to_devptr(uintptr_t p) {
    return static_cast<CUdeviceptr>(p);
}

}  // namespace

extern "C" {

sunny_geneig_solver_t sunny_geneig_create(
    int                         n,
    int                         device_id,
    int                         batches_per_block_request,
    sunny_geneig_tuning_mode_t  tuning_mode,
    sunny_geneig_cache_mode_t   cache_mode)
{
    clear_last_error();
    try {
        const TuningMode tm = (tuning_mode == SUNNY_GENEIG_TUNING_AUTOTUNE)
                                ? TuningMode::Autotune
                                : TuningMode::Suggested;
        CacheMode cm = CacheMode::Auto;
        switch (cache_mode) {
            case SUNNY_GENEIG_CACHE_NODISK:  cm = CacheMode::NoDisk;  break;
            case SUNNY_GENEIG_CACHE_NOCACHE: cm = CacheMode::NoCache; break;
            case SUNNY_GENEIG_CACHE_AUTO:
            default:                          cm = CacheMode::Auto;    break;
        }
        return new sunny_geneig_solver(n, device_id,
                                        batches_per_block_request, tm, cm);
    } catch (const std::bad_alloc& e) {
        set_last_error("out of memory in sunny_geneig_create");
        return nullptr;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("unknown C++ exception in sunny_geneig_create");
        return nullptr;
    }
}

sunny_geneig_solver_t sunny_geneig_create_ex(
    int                         n,
    int                         device_id,
    int                         batches_per_block_request,
    sunny_geneig_tuning_mode_t  tuning_mode,
    sunny_geneig_cache_mode_t   cache_mode,
    sunny_geneig_kernel_mode_t  kernel_mode)
{
    clear_last_error();
    if (kernel_mode == SUNNY_GENEIG_KERNEL_UNIFIED) {
        return sunny_geneig_create(n, device_id,
                                    batches_per_block_request,
                                    tuning_mode, cache_mode);
    }
    // SUNNY_GENEIG_KERNEL_HYBRID
    try {
        return new sunny_geneig_solver(n, device_id,
                                        SUNNY_GENEIG_KERNEL_HYBRID);
    } catch (const std::bad_alloc&) {
        set_last_error("out of memory in sunny_geneig_create_ex");
        return nullptr;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("unknown C++ exception in sunny_geneig_create_ex");
        return nullptr;
    }
}

void sunny_geneig_destroy(sunny_geneig_solver_t solver) {
    try {
        delete solver;
    } catch (...) {
        // destructors should not throw; swallow to keep ABI nothrow.
    }
}

sunny_geneig_status_t sunny_geneig_launch(
    sunny_geneig_solver_t solver,
    uintptr_t             d_H,
    uintptr_t             d_S,
    uintptr_t             d_W,
    uintptr_t             d_U,
    uintptr_t             d_info,
    int                   batch_size,
    uintptr_t             stream)
{
    clear_last_error();
    if (!solver) {
        set_last_error("sunny_geneig_launch: solver is NULL");
        return SUNNY_GENEIG_INVALID_ARG;
    }
    if (batch_size < 0) {
        set_last_error("sunny_geneig_launch: batch_size < 0");
        return SUNNY_GENEIG_INVALID_ARG;
    }
    try {
        if (solver->mode == SUNNY_GENEIG_KERNEL_HYBRID) {
            solver->hybrid->launch(
                to_devptr(d_H), to_devptr(d_S),
                to_devptr(d_W), to_devptr(d_U), to_devptr(d_info),
                batch_size, stream_from_uintptr(stream));
        } else {
            solver->unified->launch(
                to_devptr(d_H), to_devptr(d_S),
                to_devptr(d_W), to_devptr(d_U), to_devptr(d_info),
                batch_size, stream_from_uintptr(stream));
        }
        return SUNNY_GENEIG_OK;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return SUNNY_GENEIG_LAUNCH_ERROR;
    } catch (...) {
        set_last_error("unknown C++ exception in sunny_geneig_launch");
        return SUNNY_GENEIG_LAUNCH_ERROR;
    }
}

int sunny_geneig_matrix_size(sunny_geneig_solver_t solver) {
    if (!solver) return -1;
    return solver->n_for_intro;
}

int sunny_geneig_batches_per_block(sunny_geneig_solver_t solver) {
    if (!solver) return -1;
    return solver->bpb_for_intro;
}

int sunny_geneig_block_dim_x(sunny_geneig_solver_t solver) {
    if (!solver) return -1;
    return solver->bdx_for_intro;
}

unsigned int sunny_geneig_shared_mem_bytes(sunny_geneig_solver_t solver) {
    if (!solver) return 0u;
    return solver->smem_for_intro;
}

const char* sunny_geneig_last_error(void) {
    return g_last_error.c_str();
}

const char* sunny_geneig_path_name(sunny_geneig_solver_t solver) {
    if (!solver) return "";
    return (solver->mode == SUNNY_GENEIG_KERNEL_HYBRID) ? "hybrid" : "unified";
}

sunny_geneig_kernel_mode_t sunny_geneig_kernel_mode(sunny_geneig_solver_t solver) {
    if (!solver) return static_cast<sunny_geneig_kernel_mode_t>(-1);
    return solver->mode;
}

}  // extern "C"
