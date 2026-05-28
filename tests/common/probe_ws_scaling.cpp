// One-shot diagnostic: query XsyevBatched and ZheevjBatched workspace at
// batch=1, 1024, 2048 to confirm linear scaling. No timing.

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
constexpr int kN = 54;

#define CUDA_CHECK(expr) do { cudaError_t r = (expr); if (r != cudaSuccess) { std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(r)); std::exit(20); } } while (0)
#define CUSOLVER_CHECK(expr) do { cusolverStatus_t r = (expr); if (r != CUSOLVER_STATUS_SUCCESS) { std::fprintf(stderr, "cuSOLVER error: %d\n", (int)r); std::exit(21); } } while (0)
}

int main() {
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    syevjInfo_t syevj_params;
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));

    cuDoubleComplex* d_A = nullptr;
    double*          d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (std::size_t)kN * kN * 2048 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_W, (std::size_t)kN * 2048 * sizeof(double)));

    int batches[] = {1, 1024, 2048};
    std::printf("N=%d, ComplexF64\n\n", kN);

    std::printf("XsyevBatched workspace scaling:\n");
    std::printf("%-8s | %-18s | %-18s | %-12s | %-14s\n",
                "batch", "ws_gpu (B)", "ws_cpu (B)", "ws_gpu (MB)", "per matrix (KB)");
    std::printf("%s\n", "------------------------------------------------------------------------------------");
    for (int b : batches) {
        std::size_t ws_gpu = 0, ws_cpu = 0;
        CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
            handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            (int64_t)kN, CUDA_C_64F, d_A, (int64_t)kN, CUDA_R_64F, d_W,
            CUDA_C_64F, &ws_gpu, &ws_cpu, (int64_t)b));
        std::printf("%-8d | %-18zu | %-18zu | %-12.3f | %-14.3f\n",
                    b, ws_gpu, ws_cpu, ws_gpu / (1024.0 * 1024.0),
                    (ws_gpu / (double)b) / 1024.0);
    }
    std::printf("\n");

    std::printf("ZheevjBatched workspace scaling (tol=1e-7, max_sweeps=15):\n");
    std::printf("%-8s | %-18s | %-12s | %-14s\n",
                "batch", "ws (cuDC elts)", "ws (MB)", "per matrix (KB)");
    std::printf("%s\n", "------------------------------------------------------------------------");
    for (int b : batches) {
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(
            handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, kN,
            d_A, kN, d_W, &lwork, syevj_params, b));
        std::size_t ws_bytes = (std::size_t)lwork * sizeof(cuDoubleComplex);
        std::printf("%-8d | %-18d | %-12.3f | %-14.3f\n",
                    b, lwork, ws_bytes / (1024.0 * 1024.0),
                    (ws_bytes / (double)b) / 1024.0);
    }

    cudaFree(d_A); cudaFree(d_W);
    cusolverDnDestroySyevjInfo(syevj_params);
    cusolverDnDestroyParams(params);
    cusolverDnDestroy(handle);
    return 0;
}
