// module cache benchmark at N=54, BPB=1.
//
// Four-phase test:
//   1. Cold disk:    clean cache dir, construct solver_1. Full NVRTC +
//                    nvJitLink. Expected to dominate the test wall time.
//   2. Warm disk:    destroy solver_1 (so the in-process weak_ptr expires),
//                    construct solver_2 with the same key. Should hit the
//                    on-disk cubin cache. Target < 500 ms.
//   3. In-process:   keep solver_2 alive, construct solver_3 with the same
//                    key. Should hit the in-process map. Target < 5 ms.
//   4. No-cache:     construct solver_4 with CacheMode::NoCache. Should
//                    bypass both layers and pay a full compile again.
//
// All four phases validate against the same n54_b8_s42_random LAPACK
// fixture and assert bit-identical eigenvalue and eigenvector output.

#include <cuComplex.h>
#include <cuda.h>

#include "unified/unified_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

using cdouble = std::complex<double>;

namespace {

constexpr std::int32_t kMagic    = 0x47454947;
constexpr std::int32_t kVersion  = 1;
constexpr int          kN        = 54;
constexpr int          kBPB      = 1;
constexpr double       kTolEig   = 1e-10;
constexpr double       kTolVec   = 1e-10;
constexpr double       kDegenGap = 1e-6;

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

struct Fixture { int B = 0; std::vector<cdouble> H, S, U; std::vector<double> W; };

Fixture load_fixture(const std::string& path, int expected_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(10); }
    auto read_pod = [&](auto& v) {
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!f) std::exit(11);
    };
    std::int32_t magic, version, N, B, seed, reserved;
    read_pod(magic); read_pod(version); read_pod(N);
    read_pod(B);     read_pod(seed);    read_pod(reserved);
    if (magic != kMagic || version != kVersion || N != expected_n) std::exit(12);
    Fixture fx; fx.B = B;
    const std::size_t mat = static_cast<std::size_t>(N) * N;
    fx.H.resize(mat * B); fx.S.resize(mat * B); fx.U.resize(mat * B);
    fx.W.resize(static_cast<std::size_t>(N) * B);
    auto read_block = [&](void* p, std::size_t bytes) {
        f.read(reinterpret_cast<char*>(p), bytes);
        if (!f) std::exit(15);
    };
    read_block(fx.H.data(), sizeof(cdouble) * fx.H.size());
    read_block(fx.S.data(), sizeof(cdouble) * fx.S.size());
    read_block(fx.W.data(), sizeof(double)  * fx.W.size());
    read_block(fx.U.data(), sizeof(cdouble) * fx.U.size());
    return fx;
}

void mul_S_v(const cdouble* S, const cdouble* v, cdouble* Sv, int n) {
    for (int row = 0; row < n; ++row) {
        cdouble acc(0.0, 0.0);
        for (int col = 0; col < n; ++col) {
            acc += S[static_cast<std::size_t>(col) * n + row] * v[col];
        }
        Sv[row] = acc;
    }
}

// rm -rf — recursive removal of a directory tree. Used only in test
// teardown so the cold-start phase actually starts from a clean slate.
void rmrf(const std::string& path) {
    DIR* d = ::opendir(path.c_str());
    if (!d) {
        // Maybe a file or doesn't exist; try unlinking either way.
        ::unlink(path.c_str());
        return;
    }
    while (auto* e = ::readdir(d)) {
        if (std::strcmp(e->d_name, ".") == 0 || std::strcmp(e->d_name, "..") == 0)
            continue;
        std::string sub = path + "/" + e->d_name;
        struct stat st;
        if (::lstat(sub.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            rmrf(sub);
        else
            ::unlink(sub.c_str());
    }
    ::closedir(d);
    ::rmdir(path.c_str());
}

struct PhaseResult {
    std::string label;
    double      construct_ms       = 0.0;
    double      acquire_module_ms  = 0.0;
    std::string cache_layer        = "";
    int         block_dim_x        = 0;
    bool        correct            = false;
    double      max_eig_rel        = 0.0;
    double      max_phase          = 0.0;
    // Hash of W and U buffers, for bit-identical cross-phase comparison.
    std::uint64_t w_hash           = 0;
    std::uint64_t u_hash           = 0;
};

std::uint64_t fnv1a64(const void* p, std::size_t n) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

void validate(UnifiedGeneigSolver& s, const Fixture& fx, PhaseResult& r) {
    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H, mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S, mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U, mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W, static_cast<std::size_t>(kN) * fx.B * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyHtoD(d_H, fx.H.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_S, fx.S.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemsetD8(d_info, 0, fx.B * sizeof(int)));
    s.launch(d_H, d_S, d_W, d_U, d_info, fx.B, nullptr);
    CU_CHECK(cuCtxSynchronize());
    std::vector<int>     info(fx.B, 0);
    std::vector<cdouble> U_g(mat * fx.B);
    std::vector<double>  W_g(static_cast<std::size_t>(kN) * fx.B);
    CU_CHECK(cuMemcpyDtoH(info.data(), d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyDtoH(W_g.data(),  d_W,    fx.B * kN * sizeof(double)));
    CU_CHECK(cuMemcpyDtoH(U_g.data(),  d_U,    mat * fx.B * sizeof(cuDoubleComplex)));
    std::vector<cdouble> Sv(kN);
    for (int b = 0; b < fx.B; ++b) {
        if (info[b] != 0) { r.correct = false; break; }
        const cdouble* S_b   = &fx.S[mat * b];
        const cdouble* U_ref = &fx.U[mat * b];
        const double*  W_ref = &fx.W[static_cast<std::size_t>(kN) * b];
        for (int i = 0; i < kN; ++i) {
            const double scale = std::max(std::abs(W_ref[i]), 1.0);
            r.max_eig_rel = std::max(r.max_eig_rel,
                                     std::abs(W_g[static_cast<std::size_t>(kN) * b + i]
                                              - W_ref[i]) / scale);
        }
        for (int i = 0; i < kN; ++i) {
            const cdouble* u_g = &U_g[mat * b + static_cast<std::size_t>(i) * kN];
            const cdouble* u_r = &U_ref[static_cast<std::size_t>(i) * kN];
            mul_S_v(S_b, u_r, Sv.data(), kN);
            cdouble inner(0.0, 0.0);
            for (int k = 0; k < kN; ++k) inner += std::conj(u_g[k]) * Sv[k];
            const double phase_err = 1.0 - std::abs(inner);
            const double gap_left  = (i > 0)        ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
            const double gap_right = (i < kN - 1)   ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap >= kDegenGap) r.max_phase = std::max(r.max_phase, phase_err);
        }
    }
    r.correct = (r.max_eig_rel < kTolEig) && (r.max_phase < kTolVec);
    r.w_hash  = fnv1a64(W_g.data(), W_g.size() * sizeof(double));
    r.u_hash  = fnv1a64(U_g.data(), U_g.size() * sizeof(cdouble));
    cuMemFree(d_H); cuMemFree(d_S); cuMemFree(d_U); cuMemFree(d_W); cuMemFree(d_info);
}

}  // namespace

int main() {
    std::printf("cache test — N=%d, BPB=%d\n", kN, kBPB);

    const std::string cache_dir = UnifiedGeneigSolver::cache_dir();
    std::printf("cache directory: %s\n", cache_dir.c_str());
    std::printf("cleaning cache for cold-start phase...\n");
    rmrf(cache_dir);

    const Fixture fx = load_fixture("tests/data/n54_b8_s42_random.bin", kN);

    // cold disk. Full NVRTC + nvJitLink.
    PhaseResult p1; p1.label = "cold disk";
    {
        auto t0 = std::chrono::steady_clock::now();
        auto s = std::make_unique<UnifiedGeneigSolver>(
            kN, /*device_id=*/0, /*bpb=*/kBPB,
            /*force_block_dim_x=*/0,
            TuningMode::Suggested,
            CacheMode::Auto);
        auto t1 = std::chrono::steady_clock::now();
        p1.construct_ms      = std::chrono::duration<double, std::milli>(t1 - t0).count();
        p1.acquire_module_ms = s->acquire_module_ms();
        p1.cache_layer       = s->cache_layer_used();
        p1.block_dim_x       = static_cast<int>(s->block_dim().x);
        validate(*s, fx, p1);
        // Solver destructed at end of scope → in-process weak_ptr expires.
    }

    // warm disk. Disk cubin should be present from phase 1.
    PhaseResult p2; p2.label = "warm disk";
    std::unique_ptr<UnifiedGeneigSolver> s2;
    {
        auto t0 = std::chrono::steady_clock::now();
        s2 = std::make_unique<UnifiedGeneigSolver>(
            kN, /*device_id=*/0, /*bpb=*/kBPB,
            /*force_block_dim_x=*/0,
            TuningMode::Suggested,
            CacheMode::Auto);
        auto t1 = std::chrono::steady_clock::now();
        p2.construct_ms      = std::chrono::duration<double, std::milli>(t1 - t0).count();
        p2.acquire_module_ms = s2->acquire_module_ms();
        p2.cache_layer       = s2->cache_layer_used();
        p2.block_dim_x       = static_cast<int>(s2->block_dim().x);
        validate(*s2, fx, p2);
    }

    // in-process. Keep s2 alive; the weak_ptr in the cache map
    // is still locked by s2's shared_ptr.
    PhaseResult p3; p3.label = "in-process";
    {
        auto t0 = std::chrono::steady_clock::now();
        auto s3 = std::make_unique<UnifiedGeneigSolver>(
            kN, /*device_id=*/0, /*bpb=*/kBPB,
            /*force_block_dim_x=*/0,
            TuningMode::Suggested,
            CacheMode::Auto);
        auto t1 = std::chrono::steady_clock::now();
        p3.construct_ms      = std::chrono::duration<double, std::milli>(t1 - t0).count();
        p3.acquire_module_ms = s3->acquire_module_ms();
        p3.cache_layer       = s3->cache_layer_used();
        p3.block_dim_x       = static_cast<int>(s3->block_dim().x);
        validate(*s3, fx, p3);
    }

    // no-cache. Force a fresh compile, bypassing both layers.
    PhaseResult p4; p4.label = "no-cache";
    {
        auto t0 = std::chrono::steady_clock::now();
        auto s = std::make_unique<UnifiedGeneigSolver>(
            kN, /*device_id=*/0, /*bpb=*/kBPB,
            /*force_block_dim_x=*/0,
            TuningMode::Suggested,
            CacheMode::NoCache);
        auto t1 = std::chrono::steady_clock::now();
        p4.construct_ms      = std::chrono::duration<double, std::milli>(t1 - t0).count();
        p4.acquire_module_ms = s->acquire_module_ms();
        p4.cache_layer       = s->cache_layer_used();
        p4.block_dim_x       = static_cast<int>(s->block_dim().x);
        validate(*s, fx, p4);
    }

    s2.reset();  // explicit teardown for clean shutdown timing

    std::printf("\n");
    std::printf("================== cache benchmark (N=%d, BPB=%d) ==================\n",
                kN, kBPB);
    std::printf("%-12s | %4s | %14s | %14s | %-12s | %s\n",
                "phase", "BDx", "construct (ms)", "acquire (ms)",
                "cache layer", "validation");
    std::printf("-------------+------+----------------+----------------+--------------+--------\n");
    auto print_phase = [](const PhaseResult& r) {
        std::printf("%-12s | %4d | %14.1f | %14.3f | %-12s | %s "
                    "(eig=%.2e, phase=%.2e)\n",
                    r.label.c_str(), r.block_dim_x,
                    r.construct_ms, r.acquire_module_ms,
                    r.cache_layer.c_str(),
                    r.correct ? "PASS" : "FAIL",
                    r.max_eig_rel, r.max_phase);
    };
    print_phase(p1);
    print_phase(p2);
    print_phase(p3);
    print_phase(p4);
    std::printf("================================================================================\n");

    // Bit-identical output across phases — same source, same options ⇒
    // deterministic cubin ⇒ same kernel output.
    const bool w_match = (p1.w_hash == p2.w_hash)
                      && (p2.w_hash == p3.w_hash)
                      && (p3.w_hash == p4.w_hash);
    const bool u_match = (p1.u_hash == p2.u_hash)
                      && (p2.u_hash == p3.u_hash)
                      && (p3.u_hash == p4.u_hash);

    std::printf("\nchecks:\n");
    std::printf("  all four phases correct (eig<%.0e, phase<%.0e):  %s\n",
                kTolEig, kTolVec,
                (p1.correct && p2.correct && p3.correct && p4.correct) ? "PASS" : "FAIL");
    std::printf("  cold compile path:                              %s\n",
                p1.cache_layer == "compile" ? "PASS (compile)"
                                             : ("FAIL (got " + p1.cache_layer + ")").c_str());
    std::printf("  warm disk path (target < 500ms):                %s "
                "(%.1f ms, layer=%s)\n",
                (p2.cache_layer == "disk" && p2.acquire_module_ms < 500.0) ? "PASS" : "FAIL",
                p2.acquire_module_ms, p2.cache_layer.c_str());
    std::printf("  in-process path (target < 5ms):                 %s "
                "(%.3f ms, layer=%s)\n",
                (p3.cache_layer == "in-process" && p3.acquire_module_ms < 5.0) ? "PASS" : "FAIL",
                p3.acquire_module_ms, p3.cache_layer.c_str());
    std::printf("  no-cache path (compile, layer=compile):         %s "
                "(layer=%s)\n",
                p4.cache_layer == "compile" ? "PASS" : "FAIL",
                p4.cache_layer.c_str());
    std::printf("  W output bit-identical across phases:           %s "
                "(%016llx)\n",
                w_match ? "PASS" : "FAIL",
                static_cast<unsigned long long>(p1.w_hash));
    std::printf("  U output bit-identical across phases:           %s "
                "(%016llx)\n",
                u_match ? "PASS" : "FAIL",
                static_cast<unsigned long long>(p1.u_hash));

    const bool all =
        p1.correct && p2.correct && p3.correct && p4.correct
        && p1.cache_layer == "compile"
        && p2.cache_layer == "disk"       && p2.acquire_module_ms < 500.0
        && p3.cache_layer == "in-process" && p3.acquire_module_ms < 5.0
        && p4.cache_layer == "compile"
        && w_match && u_match;
    return all ? 0 : 1;
}
