// diagnostic: per-stage cost breakdown of our fused kernel.
//
// Compiles three variants of the kernel via direct NVRTC (bypassing the
// production solver and its cache) and times each on the same 1024-matrix
// workload at N=54, BlockDim=96, BPB=1 (the autotune-picked production
// configuration).
//
//   variant 0 (full):       potrf + trsm × 2 + heev<overwrite_vectors> + back-trsm
//   variant 1 (reduce):     potrf + trsm × 2                          (skip heev + back-trsm)
//   variant 2 (no_vectors): potrf + trsm × 2 + heev<no_vectors>       (skip back-trsm)
//
// Derived metrics:
//   T_heev_plus_backtrsm  = T_full - T_reduce         (Q1: heev fraction)
//   T_eigvec_cost         = T_full - T_no_vectors     (Q2: cost of computing
//                                                        eigenvectors over only
//                                                        eigenvalues)
//
// Outputs are intentionally not validated — variants 1 and 2 produce
// undefined U/W contents. The point is wall-clock timing.

#include <cuComplex.h>
#include <cuda.h>
#include <nvrtc.h>
#include <nvJitLink.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <sstream>
#include <stdexcept>
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

// Variant-aware kernel source. Differs from production kernel_source.hpp only
// in the GENEIG_VARIANT-controlled #if guards around the heev and back-trsm
// stages, plus the Heev::Job parameter.
const char* kKernelSource = R"kernel(
#include <cusolverdx.hpp>
using namespace cusolverdx;

#ifndef GENEIG_VARIANT
#define GENEIG_VARIANT 0
#endif

using Cholesky = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>()
                        + Type<type::complex>()
                        + Function<function::potrf>()
                        + FillMode<lower>()
                        + Arrangement<col_major>()
                        + LeadingDimension<SOLVER_LDA>()
                        + SM<SOLVER_SM>()
                        + Block()
                        + BlockDim<BLOCK_DIM_X>()
                        + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmLeft = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>()
                        + Type<type::complex>()
                        + Function<function::trsm>()
                        + Side<side::left>()
                        + FillMode<lower>()
                        + TransposeMode<non_trans>()
                        + Diag<diag::non_unit>()
                        + Arrangement<col_major, col_major>()
                        + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                        + SM<SOLVER_SM>()
                        + Block()
                        + BlockDim<BLOCK_DIM_X>()
                        + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmRight = decltype(Size<M_SIZE, M_SIZE>()
                         + Precision<double>()
                         + Type<type::complex>()
                         + Function<function::trsm>()
                         + Side<side::right>()
                         + FillMode<lower>()
                         + TransposeMode<conj_trans>()
                         + Diag<diag::non_unit>()
                         + Arrangement<col_major, col_major>()
                         + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                         + SM<SOLVER_SM>()
                         + Block()
                         + BlockDim<BLOCK_DIM_X>()
                         + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmLeftConj = decltype(Size<M_SIZE, M_SIZE>()
                            + Precision<double>()
                            + Type<type::complex>()
                            + Function<function::trsm>()
                            + Side<side::left>()
                            + FillMode<lower>()
                            + TransposeMode<conj_trans>()
                            + Diag<diag::non_unit>()
                            + Arrangement<col_major, col_major>()
                            + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                            + SM<SOLVER_SM>()
                            + Block()
                            + BlockDim<BLOCK_DIM_X>()
                            + BatchesPerBlock<BATCHES_PER_BLOCK>());

#if GENEIG_VARIANT == 2
using Heev = decltype(Size<M_SIZE>()
                    + Precision<double>()
                    + Type<type::complex>()
                    + Function<function::heev>()
                    + FillMode<lower>()
                    + Arrangement<col_major>()
                    + LeadingDimension<SOLVER_LDA>()
                    + Job<job::no_vectors>()
                    + SM<SOLVER_SM>()
                    + Block()
                    + BlockDim<BLOCK_DIM_X>()
                    + BatchesPerBlock<BATCHES_PER_BLOCK>());
#else
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
#endif

using DType      = typename Cholesky::a_data_type;
using PType      = typename Heev::a_precision;

constexpr int          kN_              = M_SIZE;
constexpr int          kLDA_            = SOLVER_LDA;
constexpr unsigned int kBPB_            = BATCHES_PER_BLOCK;
constexpr unsigned int kFullSmem_ =
    Cholesky::shared_memory_size + Heev::shared_memory_size;

__constant__ dim3         solver_block_dim          = Cholesky::block_dim;
__constant__ unsigned int solver_shared_memory_size = kFullSmem_;
__constant__ unsigned int solver_batches_per_block  = kBPB_;

extern "C" __global__ __launch_bounds__(Cholesky::max_threads_per_block)
void geneig_full_kernel(const cuDoubleComplex* __restrict__ H_in,
                        const cuDoubleComplex* __restrict__ S_in,
                        double*                __restrict__ W_out,
                        cuDoubleComplex*       __restrict__ U_out,
                        int*                   __restrict__ info_out,
                        int                                 batch_size) {
    const int batch_idx = blockIdx.x * kBPB_;
    if (batch_idx >= batch_size) return;

    const unsigned long long mat_stride = (unsigned long long)kN_ * kLDA_;
    const unsigned long long w_stride   = (unsigned long long)kN_;

    const cuDoubleComplex* H_b   = H_in   + mat_stride * batch_idx;
    const cuDoubleComplex* S_b   = S_in   + mat_stride * batch_idx;
    cuDoubleComplex*       U_b   = U_out  + mat_stride * batch_idx;
    double*                W_b   = W_out  + w_stride   * batch_idx;
    int*                   info_b = info_out + batch_idx;

    extern __shared__ __align__(16) unsigned char smem_raw[];

    constexpr unsigned int kBsOffset       = Cholesky::shared_memory_size;
    constexpr unsigned int kBsBytes        = sizeof(DType) * kN_ * kLDA_ * kBPB_;
    constexpr unsigned int kLambdaOffset   = kBsOffset + kBsBytes;
    constexpr unsigned int kLambdaBytes    = sizeof(PType) * kN_ * kBPB_;
    constexpr unsigned int kWorkspaceOffset =
        ((kLambdaOffset + kLambdaBytes) + alignof(DType) - 1) & ~(alignof(DType) - 1);

    DType* As           = reinterpret_cast<DType*>(smem_raw);
    DType* Bs           = reinterpret_cast<DType*>(smem_raw + kBsOffset);
    PType* lambda_s     = reinterpret_cast<PType*>(smem_raw + kLambdaOffset);
    DType* workspace_s  = reinterpret_cast<DType*>(smem_raw + kWorkspaceOffset);

    const DType* S_typed = reinterpret_cast<const DType*>(S_b);
    const DType* H_typed = reinterpret_cast<const DType*>(H_b);
    const int total_elems = kN_ * kLDA_ * (int)kBPB_;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    Cholesky().execute(As, info_b);
    __syncthreads();
    TrsmLeft().execute(As, kLDA_, Bs, kLDA_);
    __syncthreads();
    TrsmRight().execute(As, kLDA_, Bs, kLDA_);
    __syncthreads();

#if GENEIG_VARIANT != 1
    Heev().execute(Bs, kLDA_, lambda_s, workspace_s, info_b);
    __syncthreads();
#endif

#if GENEIG_VARIANT == 0
    TrsmLeftConj().execute(As, kLDA_, Bs, kLDA_);
    __syncthreads();
#endif

    // Writeback unconditional so global-memory traffic is identical across
    // variants. Output validity differs (variants 1 and 2 produce garbage in
    // U/W) but timing fairness requires equal writeback cost.
    const int total_w = kN_ * (int)kBPB_;
    for (int idx = threadIdx.x; idx < total_w; idx += blockDim.x) {
        W_b[idx] = lambda_s[idx];
    }
    DType* U_typed = reinterpret_cast<DType*>(U_b);
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        U_typed[idx] = Bs[idx];
    }
}
)kernel";

// NVRTC + nvJitLink → cubin.
std::vector<char> compile_variant(int variant) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    std::string m_size_def    = "-DM_SIZE="            + std::to_string(kN);
    std::string lda_def       = "-DSOLVER_LDA="        + std::to_string(kLDA);
    std::string sm_def        = "-DSOLVER_SM="         + std::to_string(kSM);
    std::string bpb_def       = "-DBATCHES_PER_BLOCK=" + std::to_string(kBPB);
    std::string block_dim_def = "-DBLOCK_DIM_X="       + std::to_string(kBlockDimX);
    std::string variant_def   = "-DGENEIG_VARIANT="    + std::to_string(variant);
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
        variant_def.c_str(),
        arch_opt.c_str(),
        cusolver_inc.c_str(),
        cutlass_inc.c_str(),
        cuda_inc.c_str(),
        cccl_inc.c_str()

    };

    nvrtcProgram prog = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, kKernelSource, "variant.cu",
                                   0, nullptr, nullptr));
    nvrtcResult res = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (res != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) nvrtcGetProgramLog(prog, &log[0]);
        nvrtcDestroyProgram(&prog);
        std::fprintf(stderr, "NVRTC compile failed (variant=%d): %s\n%s\n",
                     variant, nvrtcGetErrorString(res), log.c_str());
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
        std::fprintf(stderr, "nvJitLinkCreate failed (variant=%d)\n", variant);
        std::exit(23);
    }
    if (nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN,
                         CUSOLVERDX_FATBIN_PATH) != NVJITLINK_SUCCESS) {
        std::fprintf(stderr, "nvJitLinkAddFile failed (variant=%d)\n", variant);
        std::exit(24);
    }
    if (nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, lto.data(), lto_size,
                         "lto") != NVJITLINK_SUCCESS) {
        std::fprintf(stderr, "nvJitLinkAddData failed (variant=%d)\n", variant);
        std::exit(25);
    }
    if (nvJitLinkComplete(linker) != NVJITLINK_SUCCESS) {
        std::size_t lsize = 0;
        nvJitLinkGetErrorLogSize(linker, &lsize);
        std::string log(lsize, '\0');
        if (lsize > 0) nvJitLinkGetErrorLog(linker, &log[0]);
        std::fprintf(stderr, "nvJitLinkComplete failed (variant=%d): %s\n",
                     variant, log.c_str());
        std::exit(26);
    }
    std::size_t cubin_size = 0;
    nvJitLinkGetLinkedCubinSize(linker, &cubin_size);
    std::vector<char> cubin(cubin_size);
    nvJitLinkGetLinkedCubin(linker, cubin.data());
    nvJitLinkDestroy(&linker);

    auto t1 = clk::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::fprintf(stderr,
        "[compile variant=%d] %.1f s; cubin=%zu bytes\n",
        variant, sec, cubin.size());
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

void make_hpd_random(int n, std::mt19937_64& rng, cdouble* S) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble> A((std::size_t)n * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[(std::size_t)j * n + i] = cdouble(g(rng), g(rng));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            cdouble acc(0.0, 0.0);
            for (int k = 0; k < n; ++k) {
                cdouble aik = A[(std::size_t)k * n + i];
                cdouble ajk = A[(std::size_t)k * n + j];
                acc += aik * std::conj(ajk);
            }
            S[(std::size_t)j * n + i] = acc;
        }
    for (int i = 0; i < n; ++i)
        S[(std::size_t)i * n + i] += cdouble((double)n, 0.0);
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Variant {
    int          variant_id;
    const char*  label;
    CUmodule     module = nullptr;
    CUfunction   kernel = nullptr;
    dim3         block_dim {0, 0, 0};
    unsigned int shared_mem_bytes = 0;
};

void load_variant(Variant& v) {
    auto cubin = compile_variant(v.variant_id);
    CU_CHECK(cuModuleLoadDataEx(&v.module, cubin.data(), 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&v.kernel, v.module, "geneig_full_kernel"));
    {
        CUdeviceptr ptr; std::size_t sz;
        CU_CHECK(cuModuleGetGlobal(&ptr, &sz, v.module, "solver_block_dim"));
        CU_CHECK(cuMemcpyDtoH(&v.block_dim, ptr, sz));
    }
    {
        CUdeviceptr ptr; std::size_t sz;
        CU_CHECK(cuModuleGetGlobal(&ptr, &sz, v.module, "solver_shared_memory_size"));
        CU_CHECK(cuMemcpyDtoH(&v.shared_mem_bytes, ptr, sz));
    }
    CU_CHECK(cuFuncSetAttribute(v.kernel,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        (int)v.shared_mem_bytes));
    std::fprintf(stderr,
        "[load variant=%d (%s)] block_dim=(%u,%u,%u), shared_mem=%u\n",
        v.variant_id, v.label,
        v.block_dim.x, v.block_dim.y, v.block_dim.z, v.shared_mem_bytes);
}

}  // namespace

int main() {
    std::printf("heev-fraction diagnostic — N=%d, BPB=%d, "
                "BlockDim=%d, batch=%d, samples=%d\n",
                kN, kBPB, kBlockDimX, kBatch, kSamples);

    CU_CHECK(cuInit(0));
    CUdevice dev; CUcontext ctx;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    // Generate inputs once.
    const std::size_t mat = (std::size_t)kN * kLDA;
    std::vector<cdouble> H_host(mat * kBatch);
    std::vector<cdouble> S_host(mat * kBatch);
    {
        std::mt19937_64 rng(kSeed);
        for (int b = 0; b < kBatch; ++b) {
            make_hermitian(kN, rng, &H_host[mat * b]);
            make_hpd_random(kN, rng, &S_host[mat * b]);
        }
    }
    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    (std::size_t)kN * kBatch * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, kBatch * sizeof(int)));
    CU_CHECK(cuMemcpyHtoD(d_H, H_host.data(), mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_S, S_host.data(), mat * kBatch * sizeof(cuDoubleComplex)));

    // Compile and load all three variants.
    Variant variants[] = {
        {0, "full          (potrf+trsm×2+heev_overwrite_vec+back-trsm)"},
        {1, "reduce-only   (potrf+trsm×2)                              "},
        {2, "no-vectors    (potrf+trsm×2+heev_no_vec)                  "},
    };
    for (auto& v : variants) load_variant(v);

    auto run_one = [&](Variant& v, int batch_size) -> double {
        CU_CHECK(cuMemsetD8(d_info, 0, batch_size * sizeof(int)));
        int local_bs = batch_size;
        void* args[] = { &d_H, &d_S, &d_W, &d_U, &d_info, &local_bs };
        unsigned grid = (unsigned)((batch_size + kBPB - 1) / kBPB);

        CUevent e0, e1;
        cuEventCreate(&e0, CU_EVENT_DEFAULT);
        cuEventCreate(&e1, CU_EVENT_DEFAULT);
        cuEventRecord(e0, /*stream=*/nullptr);
        CU_CHECK(cuLaunchKernel(v.kernel, grid, 1, 1,
                                v.block_dim.x, v.block_dim.y, v.block_dim.z,
                                v.shared_mem_bytes, /*stream=*/nullptr,
                                args, nullptr));
        cuEventRecord(e1, /*stream=*/nullptr);
        CU_CHECK(cuCtxSynchronize());
        float ms = 0.f;
        cuEventElapsedTime(&ms, e0, e1);
        cuEventDestroy(e0); cuEventDestroy(e1);
        return (double)ms / 1000.0;
    };

    // Warm-up each variant once.
    for (auto& v : variants) run_one(v, kBatch);

    // 5-sample timing per variant.
    struct Result {
        std::string label;
        std::vector<double> samples;
        double median = 0;
        double min = 0, max = 0;
        double mps = 0;
        unsigned smem = 0;
    };
    std::vector<Result> results;
    for (auto& v : variants) {
        Result r; r.label = v.label; r.smem = v.shared_mem_bytes;
        for (int s = 0; s < kSamples; ++s) {
            r.samples.push_back(run_one(v, kBatch));
        }
        r.median = median_of(r.samples);
        r.min = *std::min_element(r.samples.begin(), r.samples.end());
        r.max = *std::max_element(r.samples.begin(), r.samples.end());
        r.mps = (double)kBatch / r.median;
        std::printf("\n--- %s ---\n", r.label.c_str());
        std::printf("  shared_mem = %u B\n", r.smem);
        std::printf("  samples: ");
        for (double t : r.samples) std::printf("%.5f ", t);
        std::printf("\n  median %.5f s, range %.5f-%.5f s, %.0f mat/sec\n",
                    r.median, r.min, r.max, r.mps);
        results.push_back(std::move(r));
    }

    const double T_full   = results[0].median;
    const double T_reduce = results[1].median;
    const double T_novec  = results[2].median;
    const double T_heev_plus_backtrsm = T_full   - T_reduce;
    const double T_eigvec_cost        = T_full   - T_novec;
    const double T_heev_only          = T_novec  - T_reduce;   // heev no-vec stage
    const double T_back_trsm_plus_vec_overhead = T_full - T_novec; // identical to T_eigvec_cost

    std::printf("\n");
    std::printf("================ per-stage breakdown (N=%d, BPB=%d, BDx=%d, batch=%d) ================\n",
                kN, kBPB, kBlockDimX, kBatch);
    std::printf("variant                     | median (s)    | mat/sec  | shared_mem | fraction of full\n");
    std::printf("----------------------------+---------------+----------+------------+-----------------\n");
    for (const auto& r : results) {
        double frac = r.median / T_full * 100.0;
        std::printf("%-27s | %13.5f | %8.0f | %10u | %6.2f%%\n",
                    r.label.c_str(), r.median, r.mps, r.smem, frac);
    }
    std::printf("=======================================================================================\n");

    std::printf("\nDerived per-stage timings (median, seconds):\n");
    std::printf("  potrf + trsmL + trsmR             : %.5f s   (%.2f%% of full)\n",
                T_reduce, 100.0 * T_reduce / T_full);
    std::printf("  heev (no-vec, eigenvalues only)   : %.5f s   (%.2f%% of full)\n",
                T_heev_only,
                100.0 * T_heev_only / T_full);
    std::printf("  heev_eigvec_overhead + back-trsm  : %.5f s   (%.2f%% of full)\n",
                T_eigvec_cost,
                100.0 * T_eigvec_cost / T_full);
    std::printf("  heev (any) + back-trsm = full - reduce : %.5f s   (%.2f%% of full)\n",
                T_heev_plus_backtrsm,
                100.0 * T_heev_plus_backtrsm / T_full);

    cuMemFree(d_H); cuMemFree(d_S); cuMemFree(d_U); cuMemFree(d_W); cuMemFree(d_info);
    for (auto& v : variants) if (v.module) cuModuleUnload(v.module);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
