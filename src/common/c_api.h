// C ABI shim around UnifiedGeneigSolver.
//
// All functions are nothrow at the C boundary. C++ exceptions inside the
// shim are converted to error codes; the most recent error message is
// available via sunny_geneig_last_error() (thread-local storage).
//
// Device pointers cross the boundary as uintptr_t — Julia's CUDA.jl
// pointers convert cleanly to UInt and we don't need to drag the CUDA
// driver header through the FFI.
//
// This is the only header Julia / external bindings need to see.

#ifndef SUNNY_GENEIG_C_API_H
#define SUNNY_GENEIG_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sunny_geneig_solver* sunny_geneig_solver_t;

typedef enum {
    SUNNY_GENEIG_OK            =  0,
    SUNNY_GENEIG_INVALID_ARG   = -1,
    SUNNY_GENEIG_DEVICE_ERROR  = -2,
    SUNNY_GENEIG_COMPILE_ERROR = -3,
    SUNNY_GENEIG_LAUNCH_ERROR  = -4
} sunny_geneig_status_t;

typedef enum {
    SUNNY_GENEIG_CACHE_AUTO    = 0,
    SUNNY_GENEIG_CACHE_NODISK  = 1,
    SUNNY_GENEIG_CACHE_NOCACHE = 2
} sunny_geneig_cache_mode_t;

typedef enum {
    SUNNY_GENEIG_TUNING_SUGGESTED = 0,
    SUNNY_GENEIG_TUNING_AUTOTUNE  = 1
} sunny_geneig_tuning_mode_t;

typedef enum {
    SUNNY_GENEIG_KERNEL_UNIFIED = 0,  // single-launch fused NVRTC kernel
    SUNNY_GENEIG_KERNEL_HYBRID  = 1   // reduce-only NVRTC + cuSOLVER XsyevBatched + cuBLAS trsm
} sunny_geneig_kernel_mode_t;

// Construct a solver. Returns NULL on failure; call sunny_geneig_last_error()
// for the message. `batches_per_block_request <= 0` means auto-select.
sunny_geneig_solver_t sunny_geneig_create(
    int                         n,
    int                         device_id,
    int                         batches_per_block_request,
    sunny_geneig_tuning_mode_t  tuning_mode,
    sunny_geneig_cache_mode_t   cache_mode);

// extended constructor adding a kernel-mode selector. The
// hybrid path (SUNNY_GENEIG_KERNEL_HYBRID) compiles a reduce-only NVRTC
// kernel and routes the eigensolve through cuSOLVER's XsyevBatched plus
// a cuBLAS back-trsm. tuning_mode and cache_mode are ignored when
// kernel_mode == SUNNY_GENEIG_KERNEL_HYBRID (the hybrid uses fixed
// BlockDim<96>, BPB=1, and no cubin cache).
sunny_geneig_solver_t sunny_geneig_create_ex(
    int                         n,
    int                         device_id,
    int                         batches_per_block_request,
    sunny_geneig_tuning_mode_t  tuning_mode,
    sunny_geneig_cache_mode_t   cache_mode,
    sunny_geneig_kernel_mode_t  kernel_mode);

// Destroy a solver. Safe to call with NULL.
void sunny_geneig_destroy(sunny_geneig_solver_t solver);

// Run the kernel. d_H, d_S are inputs (column-major Hermitian); d_W
// receives ascending eigenvalues; d_U receives generalized eigenvectors;
// d_info[i] = 0 on success per matrix. `stream` may be 0 for default.
sunny_geneig_status_t sunny_geneig_launch(
    sunny_geneig_solver_t solver,
    uintptr_t             d_H,
    uintptr_t             d_S,
    uintptr_t             d_W,
    uintptr_t             d_U,
    uintptr_t             d_info,
    int                   batch_size,
    uintptr_t             stream);

// Introspection. All return -1 / 0 if `solver` is NULL.
int          sunny_geneig_matrix_size      (sunny_geneig_solver_t solver);
int          sunny_geneig_batches_per_block(sunny_geneig_solver_t solver);
int          sunny_geneig_block_dim_x      (sunny_geneig_solver_t solver);
unsigned int sunny_geneig_shared_mem_bytes (sunny_geneig_solver_t solver);

// Most-recent error message (thread-local). Never NULL; returns "" if no
// error has been recorded on this thread.
const char*  sunny_geneig_last_error(void);

// path introspection. `sunny_geneig_path_name` returns
// "unified" or "hybrid" (or "" if solver is NULL).
// `sunny_geneig_kernel_mode` returns the SUNNY_GENEIG_KERNEL_* enum value.
const char*                sunny_geneig_path_name  (sunny_geneig_solver_t solver);
sunny_geneig_kernel_mode_t sunny_geneig_kernel_mode(sunny_geneig_solver_t solver);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SUNNY_GENEIG_C_API_H
