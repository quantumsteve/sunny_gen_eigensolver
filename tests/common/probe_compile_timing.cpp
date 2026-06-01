// probe-compile timing measurement.
//
// Question: how long does an NVRTC + nvJitLink compile take for a kernel
// source that only defines the cuSolverDx solver types and exposes their
// __constant__ suggested_block_dim values, without ever calling
// `execute()`? If fast (≤ a few seconds), we can use a two-phase compile
// strategy in UnifiedGeneigSolver: probe to read the recommendations, then
// recompile the full kernel with the chosen BlockDim.

#include <cuda.h>
#include <nvJitLink.h>
#include <nvrtc.h>
#include <vector_types.h>  // dim3
#include <algorithm>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define NVRTC_CHECK(expr) do {                                       \
    nvrtcResult _r = (expr);                                         \
    if (_r != NVRTC_SUCCESS) {                                       \
        std::fprintf(stderr, "NVRTC error at %s: %s\n",              \
                     #expr, nvrtcGetErrorString(_r));                \
        std::exit(1);                                                \
    }                                                                \
} while (0)

#define CU_CHECK(expr) do {                                          \
    CUresult _r = (expr);                                            \
    if (_r != CUDA_SUCCESS) {                                        \
        const char* _msg = nullptr;                                  \
        cuGetErrorString(_r, &_msg);                                 \
        std::fprintf(stderr, "CUDA driver error at %s: %s\n",        \
                     #expr, _msg ? _msg : "<no msg>");               \
        std::exit(2);                                                \
    }                                                                \
} while (0)

#define NVJITLINK_CHECK(linker, expr) do {                                  \
    nvJitLinkResult _r = (expr);                                            \
    if (_r != NVJITLINK_SUCCESS) {                                          \
        std::fprintf(stderr, "nvJitLink error at %s: %d\n", #expr, (int)_r);\
        std::exit(3);                                                       \
    }                                                                       \
} while (0)

// Probe kernel: defines the five solver types we use in the unified pipeline
// and exposes their `suggested_block_dim` and `block_dim` as __constant__.
// No `execute()` calls. The empty `__global__ probe_kernel` is just there to
// give nvJitLink a kernel symbol to anchor on.
const char* kProbeSource = R"kernel(
#include <cusolverdx.hpp>
using namespace cusolverdx;

using Cholesky = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>() + Type<type::complex>()
                        + Function<function::potrf>() + FillMode<lower>()
                        + Arrangement<col_major>() + LeadingDimension<SOLVER_LDA>()
                        + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                        + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmLeft = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>() + Type<type::complex>()
                        + Function<function::trsm>() + Side<side::left>()
                        + FillMode<lower>() + TransposeMode<non_trans>()
                        + Diag<diag::non_unit>()
                        + Arrangement<col_major, col_major>()
                        + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                        + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                        + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmRight = decltype(Size<M_SIZE, M_SIZE>()
                         + Precision<double>() + Type<type::complex>()
                         + Function<function::trsm>() + Side<side::right>()
                         + FillMode<lower>() + TransposeMode<conj_trans>()
                         + Diag<diag::non_unit>()
                         + Arrangement<col_major, col_major>()
                         + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                         + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                         + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmLeftConj = decltype(Size<M_SIZE, M_SIZE>()
                            + Precision<double>() + Type<type::complex>()
                            + Function<function::trsm>() + Side<side::left>()
                            + FillMode<lower>() + TransposeMode<conj_trans>()
                            + Diag<diag::non_unit>()
                            + Arrangement<col_major, col_major>()
                            + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                            + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                            + BatchesPerBlock<BATCHES_PER_BLOCK>());

using Heev = decltype(Size<M_SIZE>()
                    + Precision<double>() + Type<type::complex>()
                    + Function<function::heev>() + FillMode<lower>()
                    + Arrangement<col_major>() + LeadingDimension<SOLVER_LDA>()
                    + Job<job::overwrite_vectors>()
                    + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                    + BatchesPerBlock<BATCHES_PER_BLOCK>());

__constant__ dim3 cholesky_suggested_block_dim     = Cholesky::suggested_block_dim;
__constant__ dim3 trsm_left_suggested_block_dim    = TrsmLeft::suggested_block_dim;
__constant__ dim3 trsm_right_suggested_block_dim   = TrsmRight::suggested_block_dim;
__constant__ dim3 trsm_lc_suggested_block_dim      = TrsmLeftConj::suggested_block_dim;
__constant__ dim3 heev_suggested_block_dim         = Heev::suggested_block_dim;

extern "C" __global__ void probe_kernel() {}
)kernel";

int main() {
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    const int arch = major * 10 + minor;

    auto t_start = clk::now();
    nvrtcProgram program = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&program, kProbeSource, "probe.cu", 0, nullptr, nullptr));

    const std::string m_size_def    = "-DM_SIZE=54";
    const std::string lda_def       = "-DSOLVER_LDA=54";
    const std::string sm_def        = "-DSOLVER_SM=800";
    const std::string bpb_def       = "-DBATCHES_PER_BLOCK=1";
    const std::string arch_opt      = "--gpu-architecture=sm_" + std::to_string(arch);
    const std::string cusolver_inc  = std::string("--include-path=") + CUSOLVERDX_INCLUDE_DIR;
    const std::string cutlass_inc   = std::string("--include-path=") + CUSOLVERDX_CUTLASS_INCLUDE_DIR;
    const std::string cuda_inc      = std::string("--include-path=") + CUDA_INCLUDE_DIR;
    const std::string cccl_inc      = std::string("--include-path=") + CCCL_INCLUDE_DIR;

    const std::vector<const char*> opts = {
        "--std=c++17", "--device-as-default-execution-space",
        "-dlto", "--relocatable-device-code=true",
        m_size_def.c_str(), lda_def.c_str(), sm_def.c_str(), bpb_def.c_str(),
        arch_opt.c_str(),
        cusolver_inc.c_str(), cutlass_inc.c_str(),
        cuda_inc.c_str(), cccl_inc.c_str()
    };

    auto t_before_compile = clk::now();
    nvrtcResult res = nvrtcCompileProgram(program, (int)opts.size(), opts.data());
    auto t_after_compile = clk::now();
    if (res != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) nvrtcGetProgramLog(program, &log[0]);
        std::fprintf(stderr, "NVRTC compile failed: %s\n%s\n",
                     nvrtcGetErrorString(res), log.c_str());
        return 4;
    }

    std::size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(program, &lto_size));
    std::vector<char> lto(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(program, lto.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&program));
    auto t_after_ltoir = clk::now();

    const std::string nvjit_arch = "-arch=sm_" + std::to_string(arch);
    const char* link_opts[] = { "-lto", nvjit_arch.c_str() };
    nvJitLinkHandle linker = nullptr;
    NVJITLINK_CHECK(nullptr, nvJitLinkCreate(&linker, 2, link_opts));
    NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, CUSOLVERDX_FATBIN_PATH));
    NVJITLINK_CHECK(linker, nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, lto.data(), lto_size, "probe_lto"));
    NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));
    auto t_after_link = clk::now();

    std::size_t cubin_size = 0;
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    std::vector<char> cubin(cubin_size);
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.data()));
    nvJitLinkDestroy(&linker);

    CUmodule mod;
    CU_CHECK(cuModuleLoadDataEx(&mod, cubin.data(), 0, nullptr, nullptr));

    auto read_dim = [&](const char* name) -> dim3 {
        CUdeviceptr ptr; std::size_t size; dim3 v{0,0,0};
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, mod, name));
        CU_CHECK(cuMemcpyDtoH(&v, ptr, size));
        return v;
    };
    const dim3 bd_chol  = read_dim("cholesky_suggested_block_dim");
    const dim3 bd_trsmL = read_dim("trsm_left_suggested_block_dim");
    const dim3 bd_trsmR = read_dim("trsm_right_suggested_block_dim");
    const dim3 bd_trsmC = read_dim("trsm_lc_suggested_block_dim");
    const dim3 bd_heev  = read_dim("heev_suggested_block_dim");
    auto t_done = clk::now();

    std::printf("=== probe compile timing (N=54, BPB=1, sm_%d) ===\n", arch);
    std::printf("  nvrtcCompile     : %8.1f ms (lto_size=%zu)\n",
                ms_since(t_before_compile, t_after_compile), lto_size);
    std::printf("  GetLTOIR         : %8.1f ms\n",
                ms_since(t_after_compile, t_after_ltoir));
    std::printf("  nvJitLink+cubin  : %8.1f ms (cubin_size=%zu)\n",
                ms_since(t_after_ltoir, t_after_link), cubin_size);
    std::printf("  module+constants : %8.1f ms\n",
                ms_since(t_after_link, t_done));
    std::printf("  TOTAL            : %8.1f ms\n",
                ms_since(t_start, t_done));
    std::printf("\nSuggested block dims (.x):\n");
    std::printf("  Cholesky      : %u\n", bd_chol.x);
    std::printf("  TrsmLeft      : %u\n", bd_trsmL.x);
    std::printf("  TrsmRight     : %u\n", bd_trsmR.x);
    std::printf("  TrsmLeftConj  : %u\n", bd_trsmC.x);
    std::printf("  Heev          : %u\n", bd_heev.x);
    const unsigned chosen = std::max({bd_chol.x, bd_trsmL.x, bd_trsmR.x, bd_trsmC.x, bd_heev.x});
    std::printf("  → max         : %u\n", chosen);

    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
