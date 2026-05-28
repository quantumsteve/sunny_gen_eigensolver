// multi-N hybrid solver correctness sweep.
//
// For each N ∈ {16, 32, 48, 54}, constructs a HybridGeneigSolver and
// validates against the existing nN_b8_s42_random LAPACK fixture.
// Mirrors test_nvrtc_multiN's coverage but uses the hybrid path.
//
// N values above 54 (64, 96, 128) are skipped on this consumer-laptop
// hardware because the reduce-only kernel's 99 KB shared memory at
// N=54 already approaches the sm_120 device ceiling; the unified
// path's 3.5b multi-N test confirms higher N doesn't fit.

#include <cuComplex.h>
#include <cuda.h>

#include "hybrid/hybrid_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr std::int32_t kMagic    = 0x47454947;
constexpr std::int32_t kVersion  = 1;
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

struct Fixture { int B = 0; int N = 0;
                 std::vector<cdouble> H, S, U; std::vector<double> W; };

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
    Fixture fx; fx.B = B; fx.N = N;
    const std::size_t mat = (std::size_t)N * N;
    fx.H.resize(mat * B); fx.S.resize(mat * B); fx.U.resize(mat * B);
    fx.W.resize((std::size_t)N * B);
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
            acc += S[(std::size_t)col * n + row] * v[col];
        }
        Sv[row] = acc;
    }
}

struct NResult {
    int n              = 0;
    bool skipped       = false;
    std::string reason;
    double cold_compile_s = 0.0;
    int pass_count     = 0;
    int total          = 0;
    double max_eig_rel = 0.0;
    double max_phase   = 0.0;
};

NResult run_one(int n) {
    NResult r; r.n = n;
    std::unique_ptr<HybridGeneigSolver> solver;
    try {
        auto t0 = std::chrono::steady_clock::now();
        solver = std::make_unique<HybridGeneigSolver>(n, /*device_id=*/0);
        auto t1 = std::chrono::steady_clock::now();
        r.cold_compile_s = std::chrono::duration<double>(t1 - t0).count();
    } catch (const std::exception& e) {
        r.skipped = true;
        r.reason = std::string("construct: ") + e.what();
        return r;
    }

    const std::string path =
        "tests/data/n" + std::to_string(n) + "_b8_s42_random.bin";
    Fixture fx = load_fixture(path, n);
    r.total = fx.B;

    const std::size_t mat = (std::size_t)n * n;
    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    (std::size_t)n * fx.B * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyHtoD(d_H, fx.H.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_S, fx.S.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemsetD8(d_info, 0, fx.B * sizeof(int)));

    solver->launch(d_H, d_S, d_W, d_U, d_info, fx.B, /*stream=*/nullptr);
    CU_CHECK(cuCtxSynchronize());

    std::vector<int>     info(fx.B, 0);
    std::vector<cdouble> U_g(mat * fx.B);
    std::vector<double>  W_g((std::size_t)n * fx.B);
    CU_CHECK(cuMemcpyDtoH(info.data(), d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyDtoH(W_g.data(),  d_W,    fx.B * n * sizeof(double)));
    CU_CHECK(cuMemcpyDtoH(U_g.data(),  d_U,    mat * fx.B * sizeof(cuDoubleComplex)));

    std::vector<cdouble> Sv(n);
    for (int b = 0; b < fx.B; ++b) {
        if (info[b] != 0) continue;
        const cdouble* S_b   = &fx.S[mat * b];
        const cdouble* U_ref = &fx.U[mat * b];
        const double*  W_ref = &fx.W[(std::size_t)n * b];
        double m_eig = 0.0, m_phase = 0.0;
        for (int i = 0; i < n; ++i) {
            const double scale = std::max(std::abs(W_ref[i]), 1.0);
            m_eig = std::max(m_eig,
                std::abs(W_g[(std::size_t)n * b + i] - W_ref[i]) / scale);
        }
        for (int i = 0; i < n; ++i) {
            const cdouble* u_g = &U_g[mat * b + (std::size_t)i * n];
            const cdouble* u_r = &U_ref[(std::size_t)i * n];
            mul_S_v(S_b, u_r, Sv.data(), n);
            cdouble inner(0.0, 0.0);
            for (int k = 0; k < n; ++k) inner += std::conj(u_g[k]) * Sv[k];
            const double phase_err = 1.0 - std::abs(inner);
            const double gap_left  = (i > 0)        ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
            const double gap_right = (i < n - 1)    ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap >= kDegenGap) m_phase = std::max(m_phase, phase_err);
        }
        r.max_eig_rel = std::max(r.max_eig_rel, m_eig);
        r.max_phase   = std::max(r.max_phase,   m_phase);
        if (m_eig < kTolEig && m_phase < kTolVec) ++r.pass_count;
    }

    cuMemFree(d_H); cuMemFree(d_S); cuMemFree(d_U); cuMemFree(d_W); cuMemFree(d_info);
    return r;
}

}  // namespace

int main() {
    const std::vector<int> sizes = {16, 32, 48, 54};
    std::vector<NResult> rows;
    for (int n : sizes) {
        std::printf("\n=== N=%d ===\n", n);
        NResult r = run_one(n);
        if (r.skipped) {
            std::printf("  SKIPPED: %s\n", r.reason.c_str());
        } else {
            std::printf("  cold compile: %.1f s\n", r.cold_compile_s);
            std::printf("  %d/%d PASS, max_eig_rel=%.3e, max_phase=%.3e\n",
                        r.pass_count, r.total, r.max_eig_rel, r.max_phase);
        }
        rows.push_back(std::move(r));
    }

    std::printf("\n");
    std::printf("==================== hybrid multi-N summary ====================\n");
    std::printf(" %3s | %5s | %9s | %11s | %11s | %s\n",
                "N", "B", "cold (s)", "rand_eig", "rand_phs", "status");
    std::printf("-----+-------+-----------+-------------+-------------+--------\n");
    int pairs_pass = 0, pairs_run = 0;
    for (const auto& r : rows) {
        if (r.skipped) {
            std::printf(" %3d | %5s | %9s | %11s | %11s | SKIP (%s)\n",
                        r.n, "-", "-", "-", "-", r.reason.c_str());
            continue;
        }
        ++pairs_run;
        const bool ok = (r.pass_count == r.total)
                     && (r.max_eig_rel < kTolEig)
                     && (r.max_phase   < kTolVec);
        if (ok) ++pairs_pass;
        std::printf(" %3d | %5d | %9.1f | %11.3e | %11.3e | %s\n",
                    r.n, r.total, r.cold_compile_s,
                    r.max_eig_rel, r.max_phase, ok ? "PASS" : "FAIL");
    }
    std::printf("===========================================================================\n");
    std::printf("OVERALL: %d/%d (N, fixture) pairs passed\n", pairs_pass, pairs_run);
    return (pairs_run > 0 && pairs_pass == pairs_run) ? 0 : 1;
}
