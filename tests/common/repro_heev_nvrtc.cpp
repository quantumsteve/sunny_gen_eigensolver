// bug repro / overlay verification.
//
// Originally written to demonstrate that cuSolverDx 25.12 + NVRTC fails to
// compile a `Function<function::heev>` specialization. ships a
// header overlay (src/common/cusolverdx_overlay/) that fixes the upstream
// declaration order. With the overlay's include-path injected FIRST, this
// repro now succeeds. Without the overlay (delete CUSOLVERDX_OVERLAY_INCLUDE_DIR
// from this target's compile defs and remove the first --include-path entry
// below), it reverts to the original failure with the 6 errors documented in
// docs/CUSOLVERDX_NVRTC_HEEV_BUG.md.
//
// Standalone host-side NVRTC driver. No dependency on the project's
// UnifiedGeneigSolver class. Mirrors the NVIDIA-shipped nvrtc_potrs.cpp
// option set. Source body is the smallest possible: just instantiate
// the Solver type and declare an empty kernel — no execute() call.

#include <cuda.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define NVRTC_CHECK(expr) do {                                                  \
    nvrtcResult _r = (expr);                                                    \
    if (_r != NVRTC_SUCCESS) {                                                  \
        std::fprintf(stderr, "NVRTC error at %s: %s\n",                         \
                     #expr, nvrtcGetErrorString(_r));                           \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

#define CU_CHECK(expr) do {                                                     \
    CUresult _r = (expr);                                                       \
    if (_r != CUDA_SUCCESS) {                                                   \
        const char* _msg = nullptr;                                             \
        cuGetErrorString(_r, &_msg);                                            \
        std::fprintf(stderr, "CUDA driver error at %s: %s\n",                   \
                     #expr, _msg ? _msg : "<no msg>");                          \
        std::exit(2);                                                           \
    }                                                                           \
} while (0)

const char* kKernelSource = R"kernel(
#include <cusolverdx.hpp>
using namespace cusolverdx;

using Solver = decltype(Size<32>()
                      + Precision<double>()
                      + Type<type::complex>()
                      + Function<function::heev>()
                      + FillMode<lower>()
                      + Arrangement<col_major>()
                      + LeadingDimension<32>()
                      + Job<job::overwrite_vectors>()
                      + SM<800>()
                      + Block()
                      + BlockDim<128>());

extern "C" __global__ void k() {
    // Force class instantiation by referencing a static member.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int dummy = Solver::shared_memory_size;
        (void)dummy;
    }
}
)kernel";

int main() {
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    int arch = major * 10 + minor;

    nvrtcProgram program = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&program, kKernelSource, "repro_heev.cu",
                                   0, nullptr, nullptr));

    const std::string arch_opt     = "--gpu-architecture=sm_" + std::to_string(arch);
    const std::string overlay_inc  = std::string("--include-path=") + CUSOLVERDX_OVERLAY_INCLUDE_DIR;
    const std::string cusolver_inc = std::string("--include-path=") + CUSOLVERDX_INCLUDE_DIR;
    const std::string cutlass_inc  = std::string("--include-path=") + CUSOLVERDX_CUTLASS_INCLUDE_DIR;
    const std::string cuda_inc     = std::string("--include-path=") + CUDA_INCLUDE_DIR;

    const std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space",
        "-dlto",
        "--relocatable-device-code=true",
        arch_opt.c_str(),
        overlay_inc.c_str(),    // overlay first — shadows upstream
        cusolver_inc.c_str(),
        cutlass_inc.c_str(),
        cuda_inc.c_str(),
    };

    nvrtcResult res = nvrtcCompileProgram(program,
                                          static_cast<int>(opts.size()),
                                          opts.data());

    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 0) nvrtcGetProgramLog(program, &log[0]);

    if (res == NVRTC_SUCCESS) {
        std::printf("REPRO HEEV: NVRTC compile SUCCEEDED on sm_%d\n", arch);
        if (!log.empty() && log != std::string(log_size, '\0')) {
            std::printf("(log was non-empty, first 200 chars):\n%.200s\n", log.c_str());
        }
        nvrtcDestroyProgram(&program);
        return 0;
    }

    std::printf("REPRO HEEV: NVRTC compile FAILED on sm_%d: %s\n",
                arch, nvrtcGetErrorString(res));
    std::printf("=== full NVRTC log (%zu bytes) ===\n%s\n=== end log ===\n",
                log_size, log.c_str());
    nvrtcDestroyProgram(&program);
    return 1;
}
