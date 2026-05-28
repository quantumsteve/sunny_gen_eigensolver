// cache corruption recovery test.
//
// 1. Construct solver A with CacheMode::Auto to populate the disk cache.
// 2. Destroy A so the in-process weak_ptr expires (forcing layer 2/3 next).
// 3. Programmatically corrupt the on-disk cubin (overwrite the ELF magic
//    so try_load_disk_'s magic check trips).
// 4. Construct solver B with the same key. Expect:
//      - solver layer reports a cache "compile" (the disk file's bad magic
//        causes try_load_disk_ to remove it and fall through);
//      - the corrupted file is replaced by a fresh valid cubin;
//      - solver B produces the same output as solver A.

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

std::uint64_t fnv1a64(const void* p, std::size_t n) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

struct RunResult {
    std::string cache_layer;
    int         block_dim_x = 0;
    bool        correct     = false;
    double      max_eig_rel = 0.0;
    double      max_phase   = 0.0;
    std::uint64_t w_hash    = 0;
    std::uint64_t u_hash    = 0;
};

RunResult run_and_validate(UnifiedGeneigSolver& s, const Fixture& fx) {
    RunResult r;
    r.cache_layer = s.cache_layer_used();
    r.block_dim_x = static_cast<int>(s.block_dim().x);

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
    return r;
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

std::size_t file_size(const std::string& path) {
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return 0;
    return static_cast<std::size_t>(st.st_size);
}

}  // namespace

int main() {
    std::printf("cache invalidation test — N=%d, BPB=%d\n", kN, kBPB);
    const Fixture fx = load_fixture("tests/data/n54_b8_s42_random.bin", kN);

    // We need the cubin path BEFORE any solver constructs so we can target
    // the file directly. Probe an arch-determined key by constructing a
    // throwaway solver in NoCache mode just to discover the chosen
    // BlockDim.x — but actually the suggested-mode chosen value at N=54
    // BPB=1 has been pinned at 128 across all phases of this project, so
    // we can build the key directly. (The cubin_path() also contains the
    // arch which we get from cudaDeviceProp.) Construct A first and read
    // the key from it.

    RunResult rA, rB;
    std::string cubin_target_path;

    // Step 1: construct solver A (full compile + disk write).
    std::printf("\n[step 1] construct solver A — should populate cache.\n");
    {
        auto sA = std::make_unique<UnifiedGeneigSolver>(
            kN, /*device_id=*/0, /*bpb=*/kBPB,
            /*force_block_dim_x=*/0,
            TuningMode::Suggested,
            CacheMode::Auto);
        rA = run_and_validate(*sA, fx);
        std::printf("solver A: cache_layer=%s, BDx=%d, eig=%.2e, phase=%.2e\n",
                    rA.cache_layer.c_str(), rA.block_dim_x,
                    rA.max_eig_rel, rA.max_phase);

        // Compute the target cubin path from the key.
        int arch_dev = 0;
        CUdevice cu_dev;
        cuDeviceGet(&cu_dev, 0);
        int major = 0, minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev);
        arch_dev = major * 10 + minor;
        ModuleCacheKey key{kN, kBPB, arch_dev, rA.block_dim_x};
        cubin_target_path = UnifiedGeneigSolver::cubin_path(key);
        std::printf("target cubin path: %s\n", cubin_target_path.c_str());
        // Solver A goes out of scope here.
    }

    // Sanity: file exists from solver A.
    if (!file_exists(cubin_target_path)) {
        std::fprintf(stderr,
            "ERROR: expected cubin file does not exist after solver A construction: %s\n",
            cubin_target_path.c_str());
        return 1;
    }
    const std::size_t pre_size = file_size(cubin_target_path);
    std::printf("cubin present: %zu bytes\n", pre_size);

    // Step 2: corrupt the cubin's first byte (overwriting the ELF magic).
    // The cache loader's magic check should trip.
    std::printf("\n[step 2] overwrite first 4 bytes of cubin with 'XXXX'.\n");
    {
        std::fstream f(cubin_target_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f) {
            std::fprintf(stderr, "cannot open cubin for corruption: %s\n",
                         cubin_target_path.c_str());
            return 2;
        }
        const char garbage[4] = { 'X', 'X', 'X', 'X' };
        f.seekp(0);
        f.write(garbage, sizeof(garbage));
        if (!f) {
            std::fprintf(stderr, "corruption write failed\n");
            return 3;
        }
    }

    // Step 3: construct solver B. Expect detection + recompile + rewrite.
    std::printf("\n[step 3] construct solver B — should detect bad magic, "
                "recompile, rewrite the cubin.\n");
    {
        auto sB = std::make_unique<UnifiedGeneigSolver>(
            kN, /*device_id=*/0, /*bpb=*/kBPB,
            /*force_block_dim_x=*/0,
            TuningMode::Suggested,
            CacheMode::Auto);
        rB = run_and_validate(*sB, fx);
        std::printf("solver B: cache_layer=%s, BDx=%d, eig=%.2e, phase=%.2e\n",
                    rB.cache_layer.c_str(), rB.block_dim_x,
                    rB.max_eig_rel, rB.max_phase);
    }

    // Step 4: verify the cubin is valid again (rewritten by solver B).
    const std::size_t post_size = file_size(cubin_target_path);
    bool magic_restored = false;
    {
        std::ifstream f(cubin_target_path, std::ios::binary);
        unsigned char head[4] = {0};
        f.read(reinterpret_cast<char*>(head), 4);
        magic_restored = (head[0] == 0x7F && head[1] == 'E'
                          && head[2] == 'L' && head[3] == 'F');
    }

    std::printf("\n");
    std::printf("================== cache invalidation summary ==================\n");
    std::printf("  solver A correct:                          %s "
                "(eig=%.2e, phase=%.2e)\n",
                rA.correct ? "PASS" : "FAIL", rA.max_eig_rel, rA.max_phase);
    std::printf("  solver A cubin written:                    %s "
                "(%zu bytes)\n",
                pre_size > 1024 ? "PASS" : "FAIL", pre_size);
    std::printf("  solver B took compile path (not disk):     %s "
                "(layer=%s)\n",
                rB.cache_layer == "compile" ? "PASS" : "FAIL",
                rB.cache_layer.c_str());
    std::printf("  solver B correct after recompile:          %s "
                "(eig=%.2e, phase=%.2e)\n",
                rB.correct ? "PASS" : "FAIL", rB.max_eig_rel, rB.max_phase);
    std::printf("  cubin ELF magic restored:                  %s "
                "(post_size=%zu bytes)\n",
                magic_restored ? "PASS" : "FAIL", post_size);
    std::printf("  A and B produce bit-identical output:      %s "
                "(W=%016llx, U=%016llx)\n",
                (rA.w_hash == rB.w_hash && rA.u_hash == rB.u_hash) ? "PASS" : "FAIL",
                static_cast<unsigned long long>(rA.w_hash),
                static_cast<unsigned long long>(rA.u_hash));
    std::printf("===========================================================================\n");

    const bool ok = rA.correct && rB.correct
                 && pre_size > 1024 && post_size > 1024
                 && rB.cache_layer == "compile"
                 && magic_restored
                 && rA.w_hash == rB.w_hash && rA.u_hash == rB.u_hash;
    return ok ? 0 : 1;
}
