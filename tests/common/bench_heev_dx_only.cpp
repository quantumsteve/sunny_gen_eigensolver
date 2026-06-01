// Steven-comparison — standalone cuSolverDx::Heev-only benchmark.
//
// NVRTC-compiles a kernel that does ONLY heev (no Cholesky, no trsm) on a
// 1024-matrix N=54 ComplexF64 Hermitian batch — same matrices as
// bench_xsyev_only and bench_zheevj_only. Uses the autotune-picked production
// configuration BlockDim=96, BPB=1.
//
// This isolates cuSolverDx::Heev's eigendecomposition cost so we can compare
// it directly against cusolverDnXsyevBatched and cusolverDnZheevjBatched.

#include <cuComplex.h>
#include <cuda.h>
#include <nvrtc.h>
#include <nvJitLink.h>
#include "mkl.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr int kN          = 54;
constexpr int kLDA        = 54;
constexpr int kBPB        = 1;
constexpr int kBlockDimX  = 96;     // autotune-picked production winner
constexpr int kSM         = 800;
constexpr int kBatch      = 1024;
constexpr int kSamples    = 5;
constexpr std::uint64_t kSeed = 0xC0FFEE0042ULL;

#define CU_CHECK(expr) do {                                          \
    CUresult _r = (expr);                                            \
    if (_r != CUDA_SUCCESS) {                                        \
        const char* _msg = nullptr;                                  \
        cuGetErrorString(_r, &_msg);                                 \
        std::fprintf(stderr, "CUDA driver error %s at %s:%d: %s\n",  \
                     #expr, __FILE__, __LINE__,                      \
                     _msg ? _msg : "<no msg>");                      \
        std::exit(20);                                               \
    }                                                                \
} while (0)

#define NVRTC_CHECK(expr) do {                                          \
    nvrtcResult _r = (expr);                                            \
    if (_r != NVRTC_SUCCESS) {                                          \
        std::fprintf(stderr, "NVRTC error at %s: %s\n", #expr,          \
                     nvrtcGetErrorString(_r));                          \
        std::exit(21);                                                  \
    }                                                                   \
} while (0)

// Heev-only kernel. Loads a Hermitian H from gmem to smem, runs Heev with
// eigenvector overwrite (matches Sunny's needs), writes eigenvalues + matrix
// (now eigenvectors) back to gmem.
const char* kKernelSource = R"kernel(
#include <cusolverdx.hpp>
using namespace cusolverdx;

using Heev = decltype(Size<M_SIZE>()
                    + Precision<double>()
                    + Type<type::complex>()
                    + Function<function::heev>()
                    + FillMode<lower>()
                    + Arrangement<col_major>()
                    + LeadingDimension<SOLVER_LDA>()
                    + Job<job::overwrite_vectors>()
                    + SM<SOLVER_SM>()
                    + Block()
                    + BlockDim<BLOCK_DIM_X>()
                    + BatchesPerBlock<BATCHES_PER_BLOCK>());

using DType = typename Heev::a_data_type;
using PType = typename Heev::a_precision;

constexpr int          kN_   = M_SIZE;
constexpr int          kLDA_ = SOLVER_LDA;
constexpr unsigned int kBPB_ = BATCHES_PER_BLOCK;
constexpr unsigned int kHeevSmem_ = Heev::shared_memory_size;

__constant__ dim3         solver_block_dim          = Heev::block_dim;
__constant__ unsigned int solver_shared_memory_size = kHeevSmem_;
__constant__ unsigned int solver_batches_per_block  = kBPB_;

extern "C" __global__ __launch_bounds__(Heev::max_threads_per_block)
void heev_only_kernel(const cuDoubleComplex* __restrict__ H_in,
                      double*                __restrict__ W_out,
                      cuDoubleComplex*       __restrict__ U_out,
                      int*                   __restrict__ info_out,
                      int                                 batch_size) {
    const int batch_idx = blockIdx.x * kBPB_;
    if (batch_idx >= batch_size) return;

    const unsigned long long mat_stride = (unsigned long long)kN_ * kLDA_;
    const unsigned long long w_stride   = (unsigned long long)kN_;

    const cuDoubleComplex* H_b   = H_in   + mat_stride * batch_idx;
    cuDoubleComplex*       U_b   = U_out  + mat_stride * batch_idx;
    double*                W_b   = W_out  + w_stride   * batch_idx;
    int*                   info_b = info_out + batch_idx;

    // Smem layout matches test_heev_fraction (proven-working production
    // layout): [matrix | eigenvalues | workspace], total = Heev::shared_memory_size.
    extern __shared__ __align__(16) unsigned char smem_raw[];

    constexpr unsigned int kMatBytes      = sizeof(DType) * kN_ * kLDA_ * kBPB_;
    constexpr unsigned int kLambdaOffset  = kMatBytes;
    constexpr unsigned int kLambdaBytes   = sizeof(PType) * kN_ * kBPB_;
    constexpr unsigned int kWorkspaceOff  =
        ((kLambdaOffset + kLambdaBytes) + alignof(DType) - 1) & ~(alignof(DType) - 1);

    DType* As           = reinterpret_cast<DType*>(smem_raw);
    PType* lambda_s     = reinterpret_cast<PType*>(smem_raw + kLambdaOffset);
    DType* workspace_s  = reinterpret_cast<DType*>(smem_raw + kWorkspaceOff);

    const DType* H_typed = reinterpret_cast<const DType*>(H_b);
    const int total_elems = kN_ * kLDA_ * (int)kBPB_;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        As[idx] = H_typed[idx];
    }
    __syncthreads();

    Heev().execute(As, kLDA_, lambda_s, workspace_s, info_b);
    __syncthreads();

    const int total_w = kN_ * (int)kBPB_;
    for (int idx = threadIdx.x; idx < total_w; idx += blockDim.x) {
        W_b[idx] = lambda_s[idx];
    }
    DType* U_typed = reinterpret_cast<DType*>(U_b);
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        U_typed[idx] = As[idx];
    }
}
)kernel";

std::vector<char> compile_kernel() {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    std::string m_size_def    = "-DM_SIZE="            + std::to_string(kN);
    std::string lda_def       = "-DSOLVER_LDA="        + std::to_string(kLDA);
    std::string sm_def        = "-DSOLVER_SM="         + std::to_string(kSM);
    std::string bpb_def       = "-DBATCHES_PER_BLOCK=" + std::to_string(kBPB);
    std::string block_dim_def = "-DBLOCK_DIM_X="       + std::to_string(kBlockDimX);

    int arch = 0;
    {
        CUdevice dev; cuDeviceGet(&dev, 0);
        int major=0, minor=0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        arch = major * 10 + minor;
    }
    std::string arch_opt = "--gpu-architecture=sm_" + std::to_string(arch);

    std::string cusolver_inc = std::string("--include-path=") + CUSOLVERDX_INCLUDE_DIR;
    std::string cutlass_inc  = std::string("--include-path=") + CUSOLVERDX_CUTLASS_INCLUDE_DIR;
    std::string cuda_inc     = std::string("--include-path=") + CUDA_INCLUDE_DIR;
    std::string cccl_inc     = std::string("--include-path=") + CCCL_INCLUDE_DIR;

    std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space",
        "-dlto",
        "--relocatable-device-code=true",
        m_size_def.c_str(),
        lda_def.c_str(),
        sm_def.c_str(),
        bpb_def.c_str(),
        block_dim_def.c_str(),
        arch_opt.c_str(),
        cusolver_inc.c_str(),
        cutlass_inc.c_str(),
        cuda_inc.c_str(),
        cccl_inc.c_str()
    };

    nvrtcProgram prog = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, kKernelSource, "heev_only.cu",
                                   0, nullptr, nullptr));
    nvrtcResult res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (res != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) nvrtcGetProgramLog(prog, &log[0]);
        nvrtcDestroyProgram(&prog);
        std::fprintf(stderr, "NVRTC compile failed: %s\n%s\n",
                     nvrtcGetErrorString(res), log.c_str());
        std::exit(22);
    }
    std::size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &lto_size));
    std::vector<char> lto(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(prog, lto.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    std::string nvjit_arch = "-arch=sm_" + std::to_string(arch);
    const char* link_opts[] = { "-lto", nvjit_arch.c_str() };
    nvJitLinkHandle linker = nullptr;
    if (nvJitLinkCreate(&linker, 2, link_opts) != NVJITLINK_SUCCESS) {
        std::fprintf(stderr, "nvJitLinkCreate failed\n");
        std::exit(23);
    }
    if (nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN,
                         CUSOLVERDX_FATBIN_PATH) != NVJITLINK_SUCCESS) {
        std::fprintf(stderr, "nvJitLinkAddFile failed\n");
        std::exit(24);
    }
    if (nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, lto.data(), lto_size,
                         "lto") != NVJITLINK_SUCCESS) {
        std::fprintf(stderr, "nvJitLinkAddData failed\n");
        std::exit(25);
    }
    if (nvJitLinkComplete(linker) != NVJITLINK_SUCCESS) {
        std::size_t lsize = 0;
        nvJitLinkGetErrorLogSize(linker, &lsize);
        std::string log(lsize, '\0');
        if (lsize > 0) nvJitLinkGetErrorLog(linker, &log[0]);
        std::fprintf(stderr, "nvJitLinkComplete failed: %s\n", log.c_str());
        std::exit(26);
    }
    std::size_t cubin_size = 0;
    nvJitLinkGetLinkedCubinSize(linker, &cubin_size);
    std::vector<char> cubin(cubin_size);
    nvJitLinkGetLinkedCubin(linker, cubin.data());
    nvJitLinkDestroy(&linker);

    auto t1 = clk::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::fprintf(stderr, "[compile] %.1f s; cubin=%zu bytes\n", sec, cubin.size());
    return cubin;
}

void make_hermitian(int n, std::mt19937_64& rng, cdouble* H) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble> A((std::size_t)n * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[(std::size_t)j * n + i] = cdouble(g(rng), g(rng));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            cdouble aij = A[(std::size_t)j * n + i];
            cdouble aji = A[(std::size_t)i * n + j];
            H[(std::size_t)j * n + i] = aij + std::conj(aji);
        }
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::vector<double> lapack_reference_eigvals(const cdouble* H, int n) {
    std::vector<cdouble> work((std::size_t)n * n);
    std::memcpy(work.data(), H, sizeof(cdouble) * n * n);
    std::vector<double> w(n);
    lapack_int info = LAPACKE_zheev(
        LAPACK_COL_MAJOR, 'N', 'L', n,
        reinterpret_cast<lapack_complex_double*>(work.data()), n,
        w.data());
    if (info != 0) {
        std::fprintf(stderr, "LAPACKE_zheev reference failed: info=%d\n", (int)info);
        std::exit(30);
    }
    return w;
}

double max_rel_eig_diff(const std::vector<double>& ref,
                        const std::vector<double>& got) {
    auto a = ref; auto b = got;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::abs(a[i] - b[i]) / std::max(std::abs(a[i]), 1.0);
        if (d > m) m = d;
    }
    return m;
}

}  // namespace

int main() {
    std::printf("Steven-comparison cuSolverDx::Heev-only standalone — "
                "N=%d, BPB=%d, BlockDim=%d, batch=%d, samples=%d\n",
                kN, kBPB, kBlockDimX, kBatch, kSamples);

    CU_CHECK(cuInit(0));
    CUdevice dev; CUcontext ctx;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    const std::size_t mat = (std::size_t)kN * kLDA;
    std::vector<cdouble> H_host(mat * kBatch);
    {
        std::mt19937_64 rng(kSeed);
        for (int b = 0; b < kBatch; ++b) {
            make_hermitian(kN, rng, &H_host[mat * b]);
        }
    }

    auto ref_eigvals = lapack_reference_eigvals(H_host.data(), kN);

    auto cubin = compile_kernel();
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CU_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "heev_only_kernel"));

    dim3 block_dim;
    unsigned int shared_mem_bytes = 0;
    {
        CUdeviceptr ptr; std::size_t sz;
        CU_CHECK(cuModuleGetGlobal(&ptr, &sz, module, "solver_block_dim"));
        CU_CHECK(cuMemcpyDtoH(&block_dim, ptr, sz));
    }
    {
        CUdeviceptr ptr; std::size_t sz;
        CU_CHECK(cuModuleGetGlobal(&ptr, &sz, module, "solver_shared_memory_size"));
        CU_CHECK(cuMemcpyDtoH(&shared_mem_bytes, ptr, sz));
    }
    CU_CHECK(cuFuncSetAttribute(kernel,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        (int)shared_mem_bytes));
    std::printf("Heev kernel: block_dim=(%u,%u,%u), shared_mem=%u B = %.2f KB\n",
                block_dim.x, block_dim.y, block_dim.z,
                shared_mem_bytes, shared_mem_bytes / 1024.0);

    CUdeviceptr d_H = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    (std::size_t)kN * kBatch * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, kBatch * sizeof(int)));
    CU_CHECK(cuMemcpyHtoD(d_H, H_host.data(),
                           mat * kBatch * sizeof(cuDoubleComplex)));

    auto run_one = [&]() -> double {
        CU_CHECK(cuMemsetD8(d_info, 0, kBatch * sizeof(int)));
        int local_bs = kBatch;
        void* args[] = { &d_H, &d_W, &d_U, &d_info, &local_bs };
        unsigned grid = (unsigned)((kBatch + kBPB - 1) / kBPB);

        CUevent e0, e1;
        cuEventCreate(&e0, CU_EVENT_DEFAULT);
        cuEventCreate(&e1, CU_EVENT_DEFAULT);
        cuEventRecord(e0, /*stream=*/nullptr);
        CU_CHECK(cuLaunchKernel(kernel, grid, 1, 1,
                                block_dim.x, block_dim.y, block_dim.z,
                                shared_mem_bytes, /*stream=*/nullptr,
                                args, nullptr));
        cuEventRecord(e1, /*stream=*/nullptr);
        CU_CHECK(cuCtxSynchronize());
        float ms = 0.f;
        cuEventElapsedTime(&ms, e0, e1);
        cuEventDestroy(e0); cuEventDestroy(e1);
        return (double)ms / 1000.0;
    };

    // Warm-up.
    run_one();

    // Correctness on batch idx 0.
    run_one();
    std::vector<double> got_eigvals_b0(kN);
    CU_CHECK(cuMemcpyDtoH(got_eigvals_b0.data(), d_W, kN * sizeof(double)));
    std::vector<int> info_host(kBatch);
    CU_CHECK(cuMemcpyDtoH(info_host.data(), d_info, kBatch * sizeof(int)));
    int n_failed = 0;
    for (int i : info_host) if (i != 0) ++n_failed;
    double rel_err = max_rel_eig_diff(ref_eigvals, got_eigvals_b0);
    std::printf("Correctness (batch idx 0 vs LAPACK ZHEEV):\n");
    std::printf("  max relative eigenvalue diff = %.3e\n", rel_err);
    std::printf("  Heev info != 0 across batch: %d / %d\n", n_failed, kBatch);

    std::vector<double> samples;
    for (int s = 0; s < kSamples; ++s) {
        samples.push_back(run_one());
    }
    double med = median_of(samples);
    double lo  = *std::min_element(samples.begin(), samples.end());
    double hi  = *std::max_element(samples.begin(), samples.end());
    double mps = (double)kBatch / med;

    std::printf("\n--- cuSolverDx::Heev (overwrite_vectors, ComplexF64, N=%d, batch=%d) ---\n",
                kN, kBatch);
    std::printf("  samples (s): ");
    for (double t : samples) std::printf("%.5f ", t);
    std::printf("\n  median %.5f s, range %.5f-%.5f s, %.0f mat/sec\n",
                med, lo, hi, mps);
    std::printf("  per-matrix: %.2f us\n", med * 1e6 / kBatch);
    std::printf("  shared_mem (per block): %u B = %.2f KB\n",
                shared_mem_bytes, shared_mem_bytes / 1024.0);

    cuMemFree(d_H); cuMemFree(d_U); cuMemFree(d_W); cuMemFree(d_info);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
