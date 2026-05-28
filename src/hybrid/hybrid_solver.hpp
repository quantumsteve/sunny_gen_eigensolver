// hybrid eigensolver.
//
// Pipeline:
//   1) Our reduce-only NVRTC kernel runs on (H, S):
//        L  := chol(S)            [in-place via cuSolverDx::Cholesky::potrf]
//        M  := L⁻¹ · H · L⁻ᴴ      [via cuSolverDx::Trsm × 2]
//      Outputs: L (n×n×B), M (n×n×B), info_potrf (B).
//   2) cuSOLVER's cusolverDnXsyevBatched on M (jobz=V, uplo=L):
//        M overwritten with eigenvectors V_M; W populated with eigenvalues.
//   3) cuBLAS's cublasZtrsmBatched_64 on (L, V_M → U):
//        U := L⁻ᴴ · V_M.
//
// Stage (1) uses our fused single-kernel design — same architectural
// advantages as the unified path for the reduction stages. Stage (2)
// hands off to the mature cuSOLVER batched eigensolver, exploiting its
// grid-stride parallelism over the batch dimension. Stage (3) hands the
// back-transform to cuBLAS, which also batches efficiently.
//
// All three calls submit to the user's stream; no host-blocking syncs
// inside launch(). info from potrf and from XsyevBatched are returned
// to the caller via a single int[batch] info buffer (potrf info OR'd
// with XsyevBatched info per matrix on the device side).

#pragma once

#include <cuda.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <memory>
#include <vector>
#include <vector_types.h>

class HybridGeneigSolver {
public:
    // Construct for matrix size `n` on `device_id`. Triggers a one-time
    // NVRTC + nvJitLink compile of the reduce-only kernel (~10 s on a
    // 5070 Ti laptop). Initializes cuSOLVER and cuBLAS handles.
    explicit HybridGeneigSolver(int n, int device_id = 0);
    ~HybridGeneigSolver();

    HybridGeneigSolver(const HybridGeneigSolver&)            = delete;
    HybridGeneigSolver& operator=(const HybridGeneigSolver&) = delete;

    // Same signature as UnifiedGeneigSolver::launch. d_H, d_S are inputs;
    // d_W receives ascending eigenvalues; d_U receives generalized
    // eigenvectors (back-transformed); d_info[i] = 0 on success.
    void launch(CUdeviceptr d_H, CUdeviceptr d_S,
                CUdeviceptr d_W, CUdeviceptr d_U, CUdeviceptr d_info,
                int batch_size, CUstream stream);

    int matrix_size()  const { return n_; }
    int block_dim_x()  const { return block_dim_x_; }
    unsigned int kernel_shared_mem_bytes() const { return kernel_smem_; }
    // Highest batch the solver has so far been configured to accept (the
    // L_buf / pointer-array allocations grow on demand).
    int max_batch_seen() const { return max_batch_seen_; }

private:
    void compile_reduce_kernel_();
    void ensure_capacity_(int batch_size);
    void ensure_xsyev_workspace_(int batch_size);

    int            n_                = 0;
    int            device_id_        = 0;
    CUcontext      context_          = nullptr;
    CUmodule       module_           = nullptr;
    CUfunction     kernel_           = nullptr;
    dim3           block_dim_        {0, 0, 0};
    int            block_dim_x_      = 0;
    unsigned int   kernel_smem_      = 0;
    int            bpb_              = 1;

    cusolverDnHandle_t  solver_handle_ = nullptr;
    cusolverDnParams_t  solver_params_ = nullptr;
    cublasHandle_t      blas_handle_   = nullptr;

    // Per-launch device storage that grows on demand.
    CUdeviceptr    L_buf_            = 0;            // n×n×B ComplexF64
    int            L_buf_capacity_   = 0;            // # matrices
    CUdeviceptr    L_ptrs_           = 0;            // CuPtr<cdouble>[B]
    CUdeviceptr    U_ptrs_           = 0;            // CuPtr<cdouble>[B]
    int            ptrs_capacity_    = 0;
    void*          xsyev_ws_gpu_     = nullptr;
    std::size_t    xsyev_ws_gpu_bytes_ = 0;
    std::vector<unsigned char> xsyev_ws_cpu_;
    int            xsyev_ws_capacity_ = 0;           // batch size workspace was sized for
    int            max_batch_seen_   = 0;
};
