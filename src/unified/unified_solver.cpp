// implementation of UnifiedGeneigSolver with BatchesPerBlock support.
//
// The constructor walks the cuSolverDx-NVRTC pipeline:
//   nvrtcCreateProgram → nvrtcCompileProgram (with -dlto) → nvrtcGetLTOIR
//   → nvJitLinkCreate → nvJitLinkAddFile(libcusolverdx.fatbin)
//   → nvJitLinkAddData(LTOIR) → nvJitLinkComplete → nvJitLinkGetLinkedCubin
//   → cuModuleLoadDataEx → cuModuleGetFunction.
//
// BatchesPerBlock auto-selection: try BPB candidates in {16, 8, 4, 2, 1}.
// For each candidate that hasn't been ruled out by a quick sanity check, we
// actually compile and read back `solver_shared_memory_size`. The first one
// whose reported size fits within (device_max_smem - 2 KB) is accepted.

#include "unified/unified_solver.hpp"
#include "unified/kernel_source.hpp"

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void throw_nvrtc(nvrtcResult r, const char* call) {
    std::ostringstream oss;
    oss << "NVRTC error at " << call << ": " << nvrtcGetErrorString(r);
    throw std::runtime_error(oss.str());
}

[[noreturn]] void throw_cu(CUresult r, const char* call) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    std::ostringstream oss;
    oss << "CUDA driver error at " << call << ": " << (msg ? msg : "<no msg>");
    throw std::runtime_error(oss.str());
}

void throw_nvjitlink(nvJitLinkHandle linker, nvJitLinkResult r, const char* call) {
    std::ostringstream oss;
    oss << "nvJitLink error at " << call << ": " << static_cast<int>(r);
    if (linker) {
        std::size_t lsize = 0;
        if (nvJitLinkGetErrorLogSize(linker, &lsize) == NVJITLINK_SUCCESS && lsize > 0) {
            std::vector<char> log(lsize);
            if (nvJitLinkGetErrorLog(linker, log.data()) == NVJITLINK_SUCCESS) {
                oss << "\n" << log.data();
            }
        }
    }
    throw std::runtime_error(oss.str());
}

#define NVRTC_CHECK(expr) do {                                           \
    nvrtcResult _r = (expr);                                             \
    if (_r != NVRTC_SUCCESS) throw_nvrtc(_r, #expr);                     \
} while (0)

#define CU_CHECK(expr) do {                                              \
    CUresult _r = (expr);                                                \
    if (_r != CUDA_SUCCESS) throw_cu(_r, #expr);                         \
} while (0)

#define NVJITLINK_CHECK(linker, expr) do {                               \
    nvJitLinkResult _r = (expr);                                         \
    if (_r != NVJITLINK_SUCCESS) throw_nvjitlink((linker), _r, #expr);   \
} while (0)

// No artificial safety margin — cuFuncSetAttribute is the real gate.
// Earlier we tried 2 KB margin which incorrectly rejected the N=54 BPB=1
// case (shared_mem=99712 vs device_max=101376) on a config that we'd
// already proven works.
constexpr int kSafetyMarginBytes = 0;

}  // namespace

namespace {

struct CompileOpts {
    std::string m_size_def;
    std::string lda_def;
    std::string sm_def;
    std::string bpb_def;
    std::string block_dim_def;
    std::string arch_opt;
    std::string cusolver_inc;
    std::string cutlass_inc;
    std::string cuda_inc;
    std::string cccl_inc;
    std::vector<const char*> ptrs;
};

CompileOpts make_opts(int n, int bpb, unsigned block_dim_x, int arch) {
    CompileOpts o;
    o.m_size_def    = "-DM_SIZE="            + std::to_string(n);
    o.lda_def       = "-DSOLVER_LDA="        + std::to_string(n);
    o.sm_def        = "-DSOLVER_SM=800";
    o.bpb_def       = "-DBATCHES_PER_BLOCK=" + std::to_string(bpb);
    o.block_dim_def = "-DBLOCK_DIM_X="       + std::to_string(block_dim_x);
    o.arch_opt      = "--gpu-architecture=sm_" + std::to_string(arch);
    o.cusolver_inc  = std::string("--include-path=") + CUSOLVERDX_INCLUDE_DIR;
    o.cutlass_inc   = std::string("--include-path=") + CUSOLVERDX_CUTLASS_INCLUDE_DIR;
    o.cuda_inc      = std::string("--include-path=") + CUDA_INCLUDE_DIR;
    o.cccl_inc      = std::string("--include-path=") + CCCL_INCLUDE_DIR;

    o.ptrs = {
        "--std=c++17",
        "--device-as-default-execution-space",
        "-dlto",
        "--relocatable-device-code=true",
        o.m_size_def.c_str(),
        o.lda_def.c_str(),
        o.sm_def.c_str(),
        o.bpb_def.c_str(),
        o.block_dim_def.c_str(),
        o.arch_opt.c_str(),
        o.cusolver_inc.c_str(),
        o.cutlass_inc.c_str(),
        o.cuda_inc.c_str(),
        o.cccl_inc.c_str()
    };
    return o;
}

// Compile NVRTC source → cubin (via -dlto + nvJitLink against cuSolverDx
// fatbin). Returns the linked cubin bytes. Throws on failure with NVRTC log.
std::vector<char> nvrtc_to_cubin(const char* source,
                                 const char* source_name,
                                 const std::vector<const char*>& opts,
                                 int arch,
                                 const char* fail_context) {
    nvrtcProgram program = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&program, source, source_name,
                                   0, nullptr, nullptr));
    const nvrtcResult res =
        nvrtcCompileProgram(program, static_cast<int>(opts.size()), opts.data());
    if (res != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) nvrtcGetProgramLog(program, &log[0]);
        nvrtcDestroyProgram(&program);
        std::ostringstream oss;
        oss << "NVRTC compile failed (" << fail_context << "): "
            << nvrtcGetErrorString(res) << "\n" << log;
        throw std::runtime_error(oss.str());
    }
    std::size_t lto_size = 0;
    NVRTC_CHECK(nvrtcGetLTOIRSize(program, &lto_size));
    std::vector<char> lto(lto_size);
    NVRTC_CHECK(nvrtcGetLTOIR(program, lto.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&program));

    const std::string nvjit_arch = "-arch=sm_" + std::to_string(arch);
    const char* link_opts[] = { "-lto", nvjit_arch.c_str() };
    nvJitLinkHandle linker = nullptr;
    NVJITLINK_CHECK(nullptr, nvJitLinkCreate(&linker, 2, link_opts));
    NVJITLINK_CHECK(linker, nvJitLinkAddFile(linker,
        NVJITLINK_INPUT_FATBIN, CUSOLVERDX_FATBIN_PATH));
    NVJITLINK_CHECK(linker, nvJitLinkAddData(linker,
        NVJITLINK_INPUT_LTOIR, lto.data(), lto_size, "lto_module"));
    NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));
    std::size_t cubin_size = 0;
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    std::vector<char> cubin(cubin_size);
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.data()));
    nvJitLinkDestroy(&linker);
    return cubin;
}

int device_arch(int device_id) {
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, device_id));
    int major = 0, minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    return major * 10 + minor;
}

}  // namespace

unsigned int UnifiedGeneigSolver::probe_suggested_block_dim_x_(int bpb) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    const int arch = device_arch(device_id_);
    // Probe uses BlockDim<128> (irrelevant — any value works to read the
    // suggested constants). The probe source has no execute() calls, so
    // nvJitLink finishes in ~1 s rather than the 60–200 s of a full compile.
    const CompileOpts opts = make_opts(n_, bpb, /*block_dim_x=*/128, arch);
    const std::vector<char> cubin = nvrtc_to_cubin(
        sunny_geneig::kProbeSource, "probe.cu", opts.ptrs, arch,
        ("probe n=" + std::to_string(n_) + " bpb=" + std::to_string(bpb)).c_str());

    CUmodule mod = nullptr;
    CU_CHECK(cuModuleLoadDataEx(&mod, cubin.data(), 0, nullptr, nullptr));
    auto read_dim_x = [&](const char* name) -> unsigned {
        CUdeviceptr ptr; std::size_t size; dim3 v{0,0,0};
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, mod, name));
        CU_CHECK(cuMemcpyDtoH(&v, ptr, size));
        return v.x;
    };
    const unsigned bd_chol  = read_dim_x("cholesky_suggested_block_dim");
    const unsigned bd_trsmL = read_dim_x("trsm_left_suggested_block_dim");
    const unsigned bd_trsmR = read_dim_x("trsm_right_suggested_block_dim");
    const unsigned bd_trsmC = read_dim_x("trsm_lc_suggested_block_dim");
    const unsigned bd_heev  = read_dim_x("heev_suggested_block_dim");
    cuModuleUnload(mod);

    const unsigned chosen = std::max({bd_chol, bd_trsmL, bd_trsmR, bd_trsmC, bd_heev});
    auto t1 = clk::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
        "[UnifiedGeneigSolver] probe (n=%d bpb=%d): %.0f ms; "
        "suggested block_dim.x — chol=%u trsmL=%u trsmR=%u trsmLC=%u heev=%u → max=%u\n",
        n_, bpb, ms, bd_chol, bd_trsmL, bd_trsmR, bd_trsmC, bd_heev, chosen);
    return chosen;
}

// split the old compile_production_ into a cubin-producing
// half and a module-loading half so the cache layer can interpose on the
// boundary. compile_production_ is retained for the autotune sweep (which
// deliberately does not route through the cache).

std::vector<char> UnifiedGeneigSolver::compile_to_cubin_(int bpb,
                                                       unsigned int block_dim_x) {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    const int arch = device_arch(device_id_);
    const CompileOpts opts = make_opts(n_, bpb, block_dim_x, arch);
    std::vector<char> cubin = nvrtc_to_cubin(
        sunny_geneig::kKernelSource, "geneig_full_kernel.cu",
        opts.ptrs, arch,
        ("production n=" + std::to_string(n_) + " bpb=" + std::to_string(bpb)
         + " bdx=" + std::to_string(block_dim_x)).c_str());
    auto t1 = clk::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
        "[UnifiedGeneigSolver] compile (n=%d bpb=%d bdx=%u): %.0f ms; "
        "cubin=%zu\n",
        n_, bpb, block_dim_x, ms, cubin.size());
    return cubin;
}

std::shared_ptr<CachedModule>
UnifiedGeneigSolver::build_module_from_cubin_(const std::vector<char>& cubin,
                                            int bpb,
                                            unsigned int block_dim_x) {
    auto m = std::make_shared<CachedModule>();
    m->n                 = n_;
    m->batches_per_block = bpb;
    m->block_dim_x       = static_cast<int>(block_dim_x);
    CU_CHECK(cuModuleLoadDataEx(&m->module, cubin.data(), 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&m->kernel, m->module, "geneig_full_kernel"));
    {
        CUdeviceptr ptr; std::size_t size;
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, m->module, "solver_block_dim"));
        if (size != sizeof(dim3))
            throw std::runtime_error("solver_block_dim has unexpected size");
        CU_CHECK(cuMemcpyDtoH(&m->block_dim, ptr, size));
    }
    {
        CUdeviceptr ptr; std::size_t size;
        CU_CHECK(cuModuleGetGlobal(&ptr, &size, m->module, "solver_shared_memory_size"));
        if (size != sizeof(unsigned int))
            throw std::runtime_error("solver_shared_memory_size has unexpected size");
        CU_CHECK(cuMemcpyDtoH(&m->shared_mem_bytes, ptr, size));
    }
    return m;
}

void UnifiedGeneigSolver::compile_production_(int bpb, unsigned int block_dim_x) {
    auto cubin = compile_to_cubin_(bpb, block_dim_x);
    auto m     = build_module_from_cubin_(cubin, bpb, block_dim_x);
    // Replace the autotune-instance handle. The previous CachedModule's
    // shared_ptr ref drops here; if no other instance referenced it, its
    // destructor runs cuModuleUnload immediately.
    module_handle_    = m;
    module_           = m->module;
    kernel_           = m->kernel;
    block_dim_        = m->block_dim;
    shared_mem_bytes_ = m->shared_mem_bytes;
    std::fprintf(stderr,
        "[UnifiedGeneigSolver] autotune-step module loaded "
        "(n=%d bpb=%d bdx=%u): shared_mem=%u\n",
        n_, bpb, block_dim_x, shared_mem_bytes_);
}

// ---------------------------------------------------------------------
// cache implementation.
// ---------------------------------------------------------------------

namespace {

// FNV-1a 64-bit. Deterministic content fingerprint, not cryptographic.
// Sufficient for cache invalidation: collisions on small inputs are
// astronomically unlikely, and the cache key already pins the
// (n, bpb, arch, block_dim_x) explicitly.
constexpr std::uint64_t kFnv64Offset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kFnv64Prime  = 0x100000001b3ULL;

std::uint64_t fnv1a64(const void* data, std::size_t n,
                      std::uint64_t seed = kFnv64Offset) {
    std::uint64_t h = seed;
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<std::uint64_t>(p[i]);
        h *= kFnv64Prime;
    }
    return h;
}

// Hash of the cuSolverDx fatbin file: first 8 bytes + the byte size. Cheap
// and good enough to detect a library swap without reading the whole file.
std::uint64_t lib_hash_cached() {
    static std::uint64_t cached = 0;
    static bool          init   = false;
    if (init) return cached;
    init = true;
    struct stat st;
    if (::stat(CUSOLVERDX_FATBIN_PATH, &st) != 0) {
        cached = 0xDEADBEEFDEADBEEFULL;
        return cached;
    }
    int fd = ::open(CUSOLVERDX_FATBIN_PATH, O_RDONLY);
    if (fd < 0) {
        cached = 0xDEADBEEFDEADBEEFULL;
        return cached;
    }
    unsigned char head[8] = {0};
    auto rd = ::read(fd, head, sizeof(head));
    ::close(fd);
    std::uint64_t h = fnv1a64(head, sizeof(head));
    const std::uint64_t sz64 = static_cast<std::uint64_t>(st.st_size);
    h = fnv1a64(&sz64, sizeof(sz64), h);
    cached = h;
    (void)rd;
    return cached;
}

std::uint64_t source_hash_cached() {
    static std::uint64_t cached = 0;
    static bool          init   = false;
    if (init) return cached;
    init = true;
    cached = fnv1a64(sunny_geneig::kKernelSource,
                     std::strlen(sunny_geneig::kKernelSource));
    return cached;
}

void mkdir_p_silent(const std::string& path) {
    // Walk path components, mkdir each. Ignore EEXIST.
    std::string cur;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '/' && i > 0) {
            cur = path.substr(0, i);
            ::mkdir(cur.c_str(), 0755);
        }
    }
    ::mkdir(path.c_str(), 0755);
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

}  // namespace

std::map<ModuleCacheKey, std::weak_ptr<CachedModule>>
    UnifiedGeneigSolver::in_process_cache_;
std::mutex UnifiedGeneigSolver::in_process_cache_mutex_;

CachedModule::~CachedModule() {
    if (module) {
        cuModuleUnload(module);
        module = nullptr;
    }
}

std::string UnifiedGeneigSolver::cache_dir() {
    if (const char* env = std::getenv("SUNNY_GENEIG_CACHE_DIR")) {
        return std::string(env);
    }
#if defined(__APPLE__)
    if (const char* home = std::getenv("HOME"))
        return std::string(home) + "/Library/Caches/sunny_geneig/cubins";
#elif defined(_WIN32)
    if (const char* lad = std::getenv("LOCALAPPDATA"))
        return std::string(lad) + "\\sunny_geneig\\cubins";
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"))
        return std::string(xdg) + "/sunny_geneig/cubins";
    if (const char* home = std::getenv("HOME"))
        return std::string(home) + "/.cache/sunny_geneig/cubins";
#endif
    return std::string("/tmp/sunny_geneig/cubins");
}

std::string UnifiedGeneigSolver::cubin_path(const ModuleCacheKey& key) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s/%d_%d_%d_%d_lib%016llx_src%016llx.cubin",
                  cache_dir().c_str(),
                  key.n, key.bpb, key.arch, key.block_dim_x,
                  static_cast<unsigned long long>(lib_hash_cached()),
                  static_cast<unsigned long long>(source_hash_cached()));
    return std::string(buf);
}

std::shared_ptr<CachedModule>
UnifiedGeneigSolver::try_load_disk_(const ModuleCacheKey& key) {
    const std::string path = cubin_path(key);
    if (!file_exists(path)) return nullptr;

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "[cache] disk open failed: %s\n", path.c_str());
        return nullptr;
    }
    const std::streamsize sz = f.tellg();
    f.seekg(0);
    if (sz < 1024) {
        std::fprintf(stderr,
            "[cache] disk file too small (%lld bytes), removing: %s\n",
            static_cast<long long>(sz), path.c_str());
        f.close();
        std::remove(path.c_str());
        return nullptr;
    }
    std::vector<char> cubin(static_cast<std::size_t>(sz));
    f.read(cubin.data(), sz);
    if (!f) {
        std::fprintf(stderr, "[cache] disk read failed: %s\n", path.c_str());
        return nullptr;
    }
    // ELF magic check (CUDA cubins are ELF objects: 0x7F 'E' 'L' 'F').
    if (!(cubin.size() >= 4
          && static_cast<unsigned char>(cubin[0]) == 0x7F
          && cubin[1] == 'E' && cubin[2] == 'L' && cubin[3] == 'F')) {
        std::fprintf(stderr,
            "[cache] disk file bad ELF magic, removing: %s\n", path.c_str());
        std::remove(path.c_str());
        return nullptr;
    }
    try {
        return build_module_from_cubin_(cubin, key.bpb,
                                         static_cast<unsigned int>(key.block_dim_x));
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "[cache] cuModuleLoadDataEx failed for cached cubin %s: %s — "
            "removing and falling back to compile.\n",
            path.c_str(), e.what());
        std::remove(path.c_str());
        return nullptr;
    }
}

void UnifiedGeneigSolver::write_disk_(const ModuleCacheKey& key,
                                    const std::vector<char>& cubin) {
    mkdir_p_silent(cache_dir());
    const std::string final_path = cubin_path(key);
    char tmp_path[768];
    std::snprintf(tmp_path, sizeof(tmp_path),
                  "%s.tmp.%d.%lld",
                  final_path.c_str(),
                  static_cast<int>(::getpid()),
                  static_cast<long long>(
                      std::chrono::steady_clock::now().time_since_epoch().count()));

    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        if (!f) {
            std::fprintf(stderr,
                "[cache] tmp open failed (cannot write disk cache): %s\n",
                tmp_path);
            return;
        }
        f.write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
        if (!f) {
            std::fprintf(stderr,
                "[cache] tmp write failed: %s\n", tmp_path);
            f.close();
            std::remove(tmp_path);
            return;
        }
    }
    if (std::rename(tmp_path, final_path.c_str()) != 0) {
        std::fprintf(stderr,
            "[cache] atomic rename failed: %s -> %s\n",
            tmp_path, final_path.c_str());
        std::remove(tmp_path);
        return;
    }
    std::fprintf(stderr,
        "[cache] disk write OK: %s (%zu bytes)\n",
        final_path.c_str(), cubin.size());
}

void UnifiedGeneigSolver::acquire_module_(int bpb, unsigned int block_dim_x,
                                         CacheMode cache_mode) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const int  arch = device_arch(device_id_);
    const ModuleCacheKey key{n_, bpb, arch, static_cast<int>(block_dim_x)};

    // Layer 1: in-process cache (skipped only by NoCache).
    if (cache_mode != CacheMode::NoCache) {
        std::lock_guard<std::mutex> lock(in_process_cache_mutex_);
        auto it = in_process_cache_.find(key);
        if (it != in_process_cache_.end()) {
            if (auto h = it->second.lock()) {
                module_handle_    = h;
                module_           = h->module;
                kernel_           = h->kernel;
                block_dim_        = h->block_dim;
                shared_mem_bytes_ = h->shared_mem_bytes;
                cache_layer_used_ = "in-process";
                std::fprintf(stderr,
                    "[cache] in-process HIT (n=%d bpb=%d arch=%d bdx=%u)\n",
                    n_, bpb, arch, block_dim_x);
                acquire_module_ms_ = std::chrono::duration<double, std::milli>(
                    clk::now() - t0).count();
                return;
            }
            // Expired weak_ptr — purge and fall through.
            in_process_cache_.erase(it);
        }
    }

    // Layer 2: on-disk cubin cache (skipped by NoDisk and NoCache).
    if (cache_mode == CacheMode::Auto) {
        if (auto h = try_load_disk_(key)) {
            module_handle_    = h;
            module_           = h->module;
            kernel_           = h->kernel;
            block_dim_        = h->block_dim;
            shared_mem_bytes_ = h->shared_mem_bytes;
            cache_layer_used_ = "disk";
            {
                std::lock_guard<std::mutex> lock(in_process_cache_mutex_);
                in_process_cache_[key] = h;
            }
            std::fprintf(stderr,
                "[cache] disk HIT %s\n", cubin_path(key).c_str());
            acquire_module_ms_ = std::chrono::duration<double, std::milli>(
                clk::now() - t0).count();
            return;
        }
    }

    // Layer 3: full compile.
    std::fprintf(stderr,
        "[cache] MISS — compiling (n=%d bpb=%d arch=%d bdx=%u)\n",
        n_, bpb, arch, block_dim_x);
    auto cubin = compile_to_cubin_(bpb, block_dim_x);
    auto h     = build_module_from_cubin_(cubin, bpb, block_dim_x);

    if (cache_mode == CacheMode::Auto) {
        write_disk_(key, cubin);
    }
    if (cache_mode != CacheMode::NoCache) {
        std::lock_guard<std::mutex> lock(in_process_cache_mutex_);
        in_process_cache_[key] = h;
    }

    module_handle_    = h;
    module_           = h->module;
    kernel_           = h->kernel;
    block_dim_        = h->block_dim;
    shared_mem_bytes_ = h->shared_mem_bytes;
    cache_layer_used_ = "compile";
    acquire_module_ms_ = std::chrono::duration<double, std::milli>(
        clk::now() - t0).count();
}

// autotune sweep helpers (anonymous namespace, kept inside the
// solver translation unit so the library has no LAPACK dependency).

namespace {

using cdouble_t = std::complex<double>;

void make_hermitian_for_tune(int n, std::mt19937_64& rng, cdouble_t* H) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble_t> A(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[static_cast<std::size_t>(j) * n + i] = cdouble_t(g(rng), g(rng));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            cdouble_t aij = A[static_cast<std::size_t>(j) * n + i];
            cdouble_t aji = A[static_cast<std::size_t>(i) * n + j];
            H[static_cast<std::size_t>(j) * n + i] = aij + std::conj(aji);
        }
}

void make_hpd_for_tune(int n, std::mt19937_64& rng, cdouble_t* S) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble_t> A(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[static_cast<std::size_t>(j) * n + i] = cdouble_t(g(rng), g(rng));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            cdouble_t acc(0.0, 0.0);
            for (int k = 0; k < n; ++k) {
                cdouble_t aik = A[static_cast<std::size_t>(k) * n + i];
                cdouble_t ajk = A[static_cast<std::size_t>(k) * n + j];
                acc += aik * std::conj(ajk);
            }
            S[static_cast<std::size_t>(j) * n + i] = acc;
        }
    for (int i = 0; i < n; ++i)
        S[static_cast<std::size_t>(i) * n + i] += cdouble_t(static_cast<double>(n), 0.0);
}

double median_5(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

std::map<std::tuple<int, int, int>, int> UnifiedGeneigSolver::autotune_cache_;

unsigned int UnifiedGeneigSolver::autotune_block_dim_x_(int bpb,
                                                       unsigned int suggested) {
    using clk = std::chrono::steady_clock;
    const auto t_sweep0 = clk::now();

    // Build candidate set: {sug-64, sug-32, sug, sug+32, sug+64, sug*2}.
    // Filter to {multiples of 32, > 0, <= 1024, >= N}. Dedup. Order
    // ascending so the table reads naturally.
    std::set<int> raw{
        static_cast<int>(suggested) - 64,
        static_cast<int>(suggested) - 32,
        static_cast<int>(suggested),
        static_cast<int>(suggested) + 32,
        static_cast<int>(suggested) + 64,
        static_cast<int>(suggested) * 2,
    };
    std::vector<int> candidates;
    for (int c : raw) {
        if (c <= 0)            continue;
        if (c > 1024)          continue;
        if ((c % 32) != 0)     continue;
        if (c < n_)            continue;
        candidates.push_back(c);
    }
    std::sort(candidates.begin(), candidates.end());

    std::fprintf(stderr,
        "[autotune] (n=%d, bpb=%d) suggested=%u, candidate set = {",
        n_, bpb, suggested);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        std::fprintf(stderr, "%s%d", (i ? ", " : ""), candidates[i]);
    }
    std::fprintf(stderr, "}\n");

    if (candidates.empty()) {
        throw std::runtime_error("autotune: no candidates after filtering");
    }

    // Tuning workload size. The 3.5e spec called for 256 matrices, but
    // empirically that is too small at this hardware to discriminate the
    // actual production winner: at 256 mats with BPB=2 we launch only
    // 128 blocks on ~50 SMs (≈2.5 blocks/SM), and the GPU is so under-
    // occupied that BlockDim=128 wins by filling SMs while the production
    // 1024-mat workload is large enough that BlockDim=64 wins on the same
    // configuration. Bumped to 1024 to match the production benchmark
    // size; characterization order now agrees with production. BPB
    // divides 1024 for all BPB ∈ {1,2,4,8,16}; we round up for safety.
    constexpr int kTuneB    = 1024;
    constexpr int kSpotN    = 8;
    constexpr double kTolEig = 1e-10;
    constexpr double kTolPhs = 1e-10;
    const int tune_b = ((kTuneB + bpb - 1) / bpb) * bpb;

    const std::size_t mat = static_cast<std::size_t>(n_) * n_;
    std::vector<cdouble_t> H_host(mat * tune_b);
    std::vector<cdouble_t> S_host(mat * tune_b);
    {
        std::mt19937_64 rng(0xA5A5A5A5A5A5A5A5ULL ^ (static_cast<std::uint64_t>(n_) << 16) ^ bpb);
        for (int b = 0; b < tune_b; ++b) {
            make_hermitian_for_tune(n_, rng, &H_host[mat * b]);
            make_hpd_for_tune    (n_, rng, &S_host[mat * b]);
        }
    }

    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    const std::size_t mat_bytes = mat * tune_b * sizeof(cuDoubleComplex);
    const std::size_t w_bytes   = static_cast<std::size_t>(n_) * tune_b * sizeof(double);
    const std::size_t i_bytes   = static_cast<std::size_t>(tune_b) * sizeof(int);
    CU_CHECK(cuMemAlloc(&d_H, mat_bytes));
    CU_CHECK(cuMemAlloc(&d_S, mat_bytes));
    CU_CHECK(cuMemAlloc(&d_U, mat_bytes));
    CU_CHECK(cuMemAlloc(&d_W, w_bytes));
    CU_CHECK(cuMemAlloc(&d_info, i_bytes));
    CU_CHECK(cuMemcpyHtoD(d_H, H_host.data(), mat_bytes));
    CU_CHECK(cuMemcpyHtoD(d_S, S_host.data(), mat_bytes));

    auto launch_tune = [&](int batch_size) {
        int local_bs = batch_size;
        void* args[] = { &d_H, &d_S, &d_W, &d_U, &d_info, &local_bs };
        const unsigned grid = static_cast<unsigned>((batch_size + bpb - 1) / bpb);
        CU_CHECK(cuLaunchKernel(kernel_, grid, 1, 1,
                                block_dim_.x, block_dim_.y, block_dim_.z,
                                shared_mem_bytes_, /*stream=*/nullptr,
                                args, nullptr));
    };

    // compile suggested as the reference. We accept its output as
    // ground truth — `Suggested` mode has been LAPACK-validated end-to-end
    // in test_nvrtc_n54 and test_nvrtc_multiN, so anchoring autotune to it
    // gives transitive LAPACK coverage without dragging LAPACK into the
    // runtime library.
    compile_production_(bpb, suggested);
    CU_CHECK(cuFuncSetAttribute(kernel_,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(shared_mem_bytes_)));
    CU_CHECK(cuMemsetD8(d_info, 0, i_bytes));
    launch_tune(tune_b);
    CU_CHECK(cuCtxSynchronize());

    // Spot indices (deterministic).
    std::vector<int> spot_idx(kSpotN);
    for (int k = 0; k < kSpotN; ++k) {
        spot_idx[k] = (k * (tune_b / kSpotN)) % tune_b;
    }
    std::vector<cdouble_t> U_ref(mat * kSpotN);
    std::vector<double>    W_ref(static_cast<std::size_t>(n_) * kSpotN);
    for (int k = 0; k < kSpotN; ++k) {
        const int idx = spot_idx[k];
        CU_CHECK(cuMemcpyDtoH(&U_ref[mat * k],
                              d_U + static_cast<std::size_t>(idx) * mat * sizeof(cuDoubleComplex),
                              mat * sizeof(cuDoubleComplex)));
        CU_CHECK(cuMemcpyDtoH(&W_ref[static_cast<std::size_t>(n_) * k],
                              d_W + static_cast<std::size_t>(idx) * n_ * sizeof(double),
                              static_cast<std::size_t>(n_) * sizeof(double)));
    }

    // S-inner-product helper (small, off the hot path).
    auto mul_S_v = [&](const cdouble_t* S, const cdouble_t* v, cdouble_t* Sv) {
        for (int row = 0; row < n_; ++row) {
            cdouble_t acc(0.0, 0.0);
            for (int col = 0; col < n_; ++col) {
                acc += S[static_cast<std::size_t>(col) * n_ + row] * v[col];
            }
            Sv[row] = acc;
        }
    };

    struct CandResult {
        int    block_dim_x = 0;
        bool   compiled    = false;
        bool   correct     = false;
        double median_mps  = 0.0;
        double min_mps     = 0.0;
        double max_mps     = 0.0;
        double max_eig_rel = 0.0;
        double max_phase   = 0.0;
        std::string skip_reason;
    };
    std::vector<CandResult> sweep;
    sweep.reserve(candidates.size());

    std::vector<cdouble_t> U_g(mat);
    std::vector<double>    W_g(n_);
    std::vector<cdouble_t> Sv(n_);

    for (int cand : candidates) {
        CandResult cr; cr.block_dim_x = cand;
        // Always recompile per candidate — including for `cand == suggested`.
        // An earlier optimization skipped the recompile when cand matched
        // suggested, on the theory that we already had it loaded from the
        // Phase-1 reference compile. But by the time the sorted candidate
        // loop reaches `suggested`, an earlier (smaller) candidate has
        // already overwritten module_/kernel_, so we'd measure THAT
        // candidate's kernel under suggested's launch parameters. The bug
        // produced two adjacent rows with identical throughput and identical
        // eig/phase — a giveaway that the same binary was being benchmarked
        // twice. Keep the compile cost; correctness is non-negotiable.
        try {
            compile_production_(bpb, cand);
        } catch (const std::exception& e) {
            cr.skip_reason = e.what();
            std::fprintf(stderr,
                "[autotune] BlockDim=%d: compile rejected (%.80s...)\n",
                cand, cr.skip_reason.c_str());
            sweep.push_back(std::move(cr));
            continue;
        }
        // cuSolverDx may accept BlockDim values whose `shared_memory_size`
        // exceeds the device's per-block opt-in ceiling — its own SOLVER_SM
        // tag has a 164 KB budget at SM<800> while sm_120 laptops cap at
        // ~101 KB. cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) is
        // where the device-level rejection surfaces. Detect that here so
        // we skip cleanly instead of crashing the sweep.
        if (static_cast<int>(shared_mem_bytes_) > device_max_smem_) {
            std::ostringstream oss;
            oss << "shared_mem=" << shared_mem_bytes_
                << " > device_max=" << device_max_smem_;
            cr.skip_reason = oss.str();
            std::fprintf(stderr,
                "[autotune] BlockDim=%d: skipped (%s)\n",
                cand, cr.skip_reason.c_str());
            sweep.push_back(std::move(cr));
            continue;
        }
        CU_CHECK(cuFuncSetAttribute(kernel_,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(shared_mem_bytes_)));
        cr.compiled = true;

        // 5-sample median timing.
        std::vector<double> mps_samples; mps_samples.reserve(5);
        for (int s = 0; s < 5; ++s) {
            CU_CHECK(cuMemsetD8(d_info, 0, i_bytes));
            CUevent e0, e1;
            cuEventCreate(&e0, CU_EVENT_DEFAULT);
            cuEventCreate(&e1, CU_EVENT_DEFAULT);
            cuEventRecord(e0, nullptr);
            launch_tune(tune_b);
            cuEventRecord(e1, nullptr);
            CU_CHECK(cuCtxSynchronize());
            float ms = 0.f;
            cuEventElapsedTime(&ms, e0, e1);
            cuEventDestroy(e0); cuEventDestroy(e1);
            mps_samples.push_back(static_cast<double>(tune_b) / (ms / 1000.0));
        }
        cr.median_mps = median_5(mps_samples);
        cr.min_mps    = *std::min_element(mps_samples.begin(), mps_samples.end());
        cr.max_mps    = *std::max_element(mps_samples.begin(), mps_samples.end());

        // Correctness validation: per-spot eigenvalue relative diff and
        // S-inner-product phase, comparing against the suggested reference.
        for (int k = 0; k < kSpotN; ++k) {
            const int idx     = spot_idx[k];
            const cdouble_t* S_b   = &S_host[mat * idx];
            const cdouble_t* U_r_k = &U_ref[mat * k];
            const double*    W_r_k = &W_ref[static_cast<std::size_t>(n_) * k];
            CU_CHECK(cuMemcpyDtoH(W_g.data(),
                                  d_W + static_cast<std::size_t>(idx) * n_ * sizeof(double),
                                  static_cast<std::size_t>(n_) * sizeof(double)));
            CU_CHECK(cuMemcpyDtoH(U_g.data(),
                                  d_U + static_cast<std::size_t>(idx) * mat * sizeof(cuDoubleComplex),
                                  mat * sizeof(cuDoubleComplex)));
            for (int i = 0; i < n_; ++i) {
                const double scale = std::max(std::abs(W_r_k[i]), 1.0);
                cr.max_eig_rel = std::max(cr.max_eig_rel,
                                          std::abs(W_g[i] - W_r_k[i]) / scale);
            }
            for (int i = 0; i < n_; ++i) {
                const cdouble_t* u_g = &U_g[static_cast<std::size_t>(i) * n_];
                const cdouble_t* u_r = &U_r_k[static_cast<std::size_t>(i) * n_];
                mul_S_v(S_b, u_r, Sv.data());
                cdouble_t inner(0.0, 0.0);
                for (int j = 0; j < n_; ++j) inner += std::conj(u_g[j]) * Sv[j];
                cr.max_phase = std::max(cr.max_phase, 1.0 - std::abs(inner));
            }
        }
        cr.correct = (cr.max_eig_rel < kTolEig) && (cr.max_phase < kTolPhs);
        std::fprintf(stderr,
            "[autotune] BlockDim=%d: median %.0f mat/s, range %.0f..%.0f, "
            "eig_rel=%.2e, phase=%.2e %s\n",
            cand, cr.median_mps, cr.min_mps, cr.max_mps,
            cr.max_eig_rel, cr.max_phase,
            cr.correct ? "[OK]" : "[REJECT correctness]");
        sweep.push_back(std::move(cr));
    }

    // Free device buffers.
    cuMemFree(d_H); cuMemFree(d_S); cuMemFree(d_U);
    cuMemFree(d_W); cuMemFree(d_info);

    // Pick winner: fastest correctness-validated candidate. Tie-break on
    // smaller BlockDim (less waste).
    int winner = -1;
    double winner_mps = -1.0;
    for (const auto& cr : sweep) {
        if (!cr.correct) continue;
        if (cr.median_mps > winner_mps) {
            winner_mps = cr.median_mps;
            winner     = cr.block_dim_x;
        }
    }
    if (winner < 0) {
        throw std::runtime_error("autotune: no correctness-validated candidate "
                                  "(all candidates either failed compile or "
                                  "produced incorrect output)");
    }

    const auto t_sweep1 = clk::now();
    const double sweep_s = std::chrono::duration<double>(t_sweep1 - t_sweep0).count();
    std::fprintf(stderr,
        "[autotune] sweep took %.1f s; winner BlockDim=%d at %.0f mat/s "
        "(suggested=%u)\n",
        sweep_s, winner, winner_mps, suggested);

    autotune_sweep_ran_ = true;
    return static_cast<unsigned int>(winner);
}

UnifiedGeneigSolver::UnifiedGeneigSolver(int n,
                                     int device_id,
                                     int batches_per_block_request,
                                     int force_block_dim_x,
                                     TuningMode tuning_mode,
                                     CacheMode  cache_mode)
    : n_(n), device_id_(device_id) {
    const TuningMode mode = tuning_mode;  // local alias preserves prior code
    CU_CHECK(cuInit(0));
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id_));
    CU_CHECK(cuDevicePrimaryCtxRetain(&context_, cu_device));
    CU_CHECK(cuCtxSetCurrent(context_));

    CU_CHECK(cuDeviceGetAttribute(&device_max_smem_,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cu_device));

    const int budget = device_max_smem_ - kSafetyMarginBytes;
    if (budget <= 0) {
        std::ostringstream oss;
        oss << "device max dynamic shared memory (" << device_max_smem_
            << " B) is below safety margin (" << kSafetyMarginBytes << " B)";
        throw std::runtime_error(oss.str());
    }

    auto try_one = [&](int bpb) -> bool {
        unsigned int chosen_bd_x = 0;
        if (force_block_dim_x > 0) {
            chosen_bd_x = static_cast<unsigned>(force_block_dim_x);
            std::fprintf(stderr,
                "[UnifiedGeneigSolver] BPB=%d forcing block_dim.x=%u "
                "(probe and autotune skipped)\n",
                bpb, chosen_bd_x);
        } else {
            unsigned int suggested = 0;
            try {
                suggested = probe_suggested_block_dim_x_(bpb);
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                    "[UnifiedGeneigSolver] BPB=%d probe rejected: %s\n",
                    bpb, e.what());
                return false;
            }
            if (mode == TuningMode::Autotune) {
                const int arch = device_arch(device_id_);
                const auto key = std::make_tuple(n_, bpb, arch);
                auto it = autotune_cache_.find(key);
                if (it != autotune_cache_.end()) {
                    chosen_bd_x = static_cast<unsigned int>(it->second);
                    std::fprintf(stderr,
                        "[autotune] cache HIT for (n=%d, bpb=%d, arch=%d) "
                        "→ BlockDim.x=%u (suggested=%u)\n",
                        n_, bpb, arch, chosen_bd_x, suggested);
                    autotune_sweep_ran_ = false;
                } else {
                    try {
                        chosen_bd_x = autotune_block_dim_x_(bpb, suggested);
                    } catch (const std::exception& e) {
                        std::fprintf(stderr,
                            "[UnifiedGeneigSolver] BPB=%d autotune failed: %s\n",
                            bpb, e.what());
                        return false;
                    }
                    autotune_cache_[key] = static_cast<int>(chosen_bd_x);
                }
                was_autotuned_         = true;
                autotuned_block_dim_x_ = static_cast<int>(chosen_bd_x);
            } else {
                chosen_bd_x = suggested;
            }
        }
        try {
            // final production module is acquired through the
            // cache layer (in-process → disk → compile fallback). The
            // autotune sweep above used the non-caching compile_production_
            // because per-candidate cubins are throwaway. Here, the winner
            // is a long-lived production artifact and benefits from caching.
            acquire_module_(bpb, chosen_bd_x, cache_mode);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "[UnifiedGeneigSolver] BPB=%d production rejected: %s\n",
                bpb, e.what());
            return false;
        }
        if (static_cast<int>(shared_mem_bytes_) > budget) {
            std::fprintf(stderr,
                "[UnifiedGeneigSolver] BPB=%d shared_mem=%u > budget=%d\n",
                bpb, shared_mem_bytes_, budget);
            return false;
        }
        batches_per_block_ = bpb;
        return true;
    };

    if (batches_per_block_request > 0) {
        if (!try_one(batches_per_block_request)) {
            std::ostringstream oss;
            oss << "explicit BPB=" << batches_per_block_request
                << " could not be constructed (NVRTC reject or shared_mem "
                << shared_mem_bytes_ << " > budget " << budget << ")";
            throw std::runtime_error(oss.str());
        }
    } else {
        const int candidates[] = {16, 8, 4, 2, 1};
        bool ok = false;
        for (int cand : candidates) {
            if (try_one(cand)) { ok = true; break; }
        }
        if (!ok) {
            std::ostringstream oss;
            oss << "no BPB candidate in {16,8,4,2,1} fits device budget "
                << budget << " B (max " << device_max_smem_ << " B). "
                << "Last attempted BPB shared_mem=" << shared_mem_bytes_;
            throw std::runtime_error(oss.str());
        }
    }

    CU_CHECK(cuFuncSetAttribute(kernel_,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(shared_mem_bytes_)));
}

UnifiedGeneigSolver::~UnifiedGeneigSolver() {
    // drop the shared_ptr. If this was the last instance
    // referencing the CachedModule, its destructor unloads the CUmodule;
    // otherwise the module lives on serving the next solver instance.
    module_handle_.reset();
    module_ = nullptr;
    kernel_ = nullptr;
    CUdevice dev;
    if (cuDeviceGet(&dev, device_id_) == CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(dev);
    }
    context_ = nullptr;
}

void UnifiedGeneigSolver::launch(CUdeviceptr d_H, CUdeviceptr d_S,
                                CUdeviceptr d_W, CUdeviceptr d_U,
                                CUdeviceptr d_info,
                                int batch_size, CUstream stream) {
    if (batch_size <= 0) return;
    int local_bs = batch_size;
    void* args[] = { &d_H, &d_S, &d_W, &d_U, &d_info, &local_bs };
    const unsigned grid =
        static_cast<unsigned>((batch_size + batches_per_block_ - 1) / batches_per_block_);
    CU_CHECK(cuLaunchKernel(kernel_,
                            /*grid:*/ grid, 1, 1,
                            /*block:*/ block_dim_.x, block_dim_.y, block_dim_.z,
                            shared_mem_bytes_, stream, args, nullptr));
}

void UnifiedGeneigSolver::launch_chunked(CUdeviceptr d_H, CUdeviceptr d_S,
                                       CUdeviceptr d_W, CUdeviceptr d_U,
                                       CUdeviceptr d_info,
                                       int total_matrices, int chunk_size,
                                       int num_streams,
                                       const std::vector<CUstream>& streams) {
    if (total_matrices <= 0) return;
    if (chunk_size <= 0) {
        throw std::runtime_error("launch_chunked: chunk_size must be > 0");
    }
    if (num_streams <= 0 || static_cast<int>(streams.size()) != num_streams) {
        throw std::runtime_error("launch_chunked: streams.size() must equal num_streams");
    }
    const std::size_t mat_elts  = static_cast<std::size_t>(n_) * n_;
    const std::size_t mat_bytes = mat_elts * sizeof(cuDoubleComplex);
    const std::size_t w_bytes_per = static_cast<std::size_t>(n_) * sizeof(double);

    int chunk_idx = 0;
    for (int offset = 0; offset < total_matrices; offset += chunk_size) {
        const int  bs        = std::min(chunk_size, total_matrices - offset);
        CUstream   s         = streams[chunk_idx % num_streams];
        const std::size_t mat_off = static_cast<std::size_t>(offset) * mat_bytes;
        const std::size_t w_off   = static_cast<std::size_t>(offset) * w_bytes_per;
        const std::size_t i_off   = static_cast<std::size_t>(offset) * sizeof(int);
        launch(d_H + mat_off, d_S + mat_off,
               d_W + w_off,   d_U + mat_off, d_info + i_off,
               bs, s);
        ++chunk_idx;
    }
}

std::vector<CUstream> make_streams(int n) {
    std::vector<CUstream> v;
    v.reserve(n);
    for (int i = 0; i < n; ++i) {
        CUstream s = nullptr;
        CU_CHECK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
        v.push_back(s);
    }
    return v;
}

void destroy_streams(std::vector<CUstream>& streams) {
    for (CUstream s : streams) {
        if (s) cuStreamDestroy(s);
    }
    streams.clear();
}
