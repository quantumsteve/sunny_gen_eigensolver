// single-N=54 hybrid solver correctness test.
//
// Mirrors test_nvrtc_n54 but constructs a HybridGeneigSolver and drives
// it on the n54_b8_s42_random.bin LAPACK fixture. 8/8 PASS at fp64
// epsilon required.

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
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr std::int32_t kMagic    = 0x47454947;
constexpr std::int32_t kVersion  = 1;
constexpr int          kN        = 54;
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

Fixture load_fixture(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(10); }
    auto read_pod = [&](auto& v) {
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!f) std::exit(11);
    };
    std::int32_t magic, version, N, B, seed, reserved;
    read_pod(magic); read_pod(version); read_pod(N);
    read_pod(B);     read_pod(seed);    read_pod(reserved);
    if (magic != kMagic || version != kVersion || N != kN) std::exit(12);
    Fixture fx; fx.B = B;
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

}  // namespace

int main() {
    auto t0 = std::chrono::steady_clock::now();
    HybridGeneigSolver solver(kN, /*device_id=*/0);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("hybrid construct (incl. NVRTC reduce-kernel compile): %.1f s\n",
                std::chrono::duration<double>(t1 - t0).count());
    std::printf("kernel block_dim_x=%d, kernel_smem=%u, matrix_size=%d\n",
                solver.block_dim_x(), solver.kernel_shared_mem_bytes(),
                solver.matrix_size());

    const Fixture fx = load_fixture("tests/data/n54_b8_s42_random.bin");
    std::printf("loaded fixture: B=%d, N=%d\n", fx.B, kN);

    const std::size_t mat = (std::size_t)kN * kN;
    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    (std::size_t)kN * fx.B * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyHtoD(d_H, fx.H.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_S, fx.S.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemsetD8(d_info, 0, fx.B * sizeof(int)));

    solver.launch(d_H, d_S, d_W, d_U, d_info, fx.B, /*stream=*/nullptr);
    CU_CHECK(cuCtxSynchronize());

    std::vector<int>     info(fx.B, 0);
    std::vector<cdouble> U_g(mat * fx.B);
    std::vector<double>  W_g((std::size_t)kN * fx.B);
    CU_CHECK(cuMemcpyDtoH(info.data(), d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyDtoH(W_g.data(),  d_W,    fx.B * kN * sizeof(double)));
    CU_CHECK(cuMemcpyDtoH(U_g.data(),  d_U,    mat * fx.B * sizeof(cuDoubleComplex)));

    int    pass_count = 0;
    double max_eig_rel_overall = 0.0;
    double max_phase_overall   = 0.0;
    std::vector<cdouble> Sv(kN);

    for (int b = 0; b < fx.B; ++b) {
        if (info[b] != 0) {
            std::printf("matrix %d: info=%d  FAIL\n", b, info[b]);
            continue;
        }
        const cdouble* S_b   = &fx.S[mat * b];
        const cdouble* U_ref = &fx.U[mat * b];
        const double*  W_ref = &fx.W[(std::size_t)kN * b];

        double max_eig_rel = 0.0;
        for (int i = 0; i < kN; ++i) {
            const double scale = std::max(std::abs(W_ref[i]), 1.0);
            max_eig_rel = std::max(max_eig_rel,
                std::abs(W_g[(std::size_t)kN * b + i] - W_ref[i]) / scale);
        }
        double max_phase = 0.0;
        int    degen_excluded = 0;
        for (int i = 0; i < kN; ++i) {
            const cdouble* u_g = &U_g[mat * b + (std::size_t)i * kN];
            const cdouble* u_r = &U_ref[(std::size_t)i * kN];
            mul_S_v(S_b, u_r, Sv.data(), kN);
            cdouble inner(0.0, 0.0);
            for (int k = 0; k < kN; ++k) inner += std::conj(u_g[k]) * Sv[k];
            const double phase_err = 1.0 - std::abs(inner);
            const double gap_left  = (i > 0)        ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
            const double gap_right = (i < kN - 1)   ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap < kDegenGap) ++degen_excluded;
            else max_phase = std::max(max_phase, phase_err);
        }
        max_eig_rel_overall = std::max(max_eig_rel_overall, max_eig_rel);
        max_phase_overall   = std::max(max_phase_overall, max_phase);
        const bool ok = (max_eig_rel < kTolEig) && (max_phase < kTolVec);
        std::printf("matrix %d: eig_rel=%.3e, phase_err=%.3e", b, max_eig_rel, max_phase);
        if (degen_excluded > 0) std::printf(" (degen_excluded=%d)", degen_excluded);
        std::printf("  %s\n", ok ? "PASS" : "FAIL");
        if (ok) ++pass_count;
    }
    std::printf("SUMMARY: %d/%d passed, max_eig_rel=%.3e, max_phase=%.3e\n",
                pass_count, fx.B, max_eig_rel_overall, max_phase_overall);

    cuMemFree(d_H); cuMemFree(d_S); cuMemFree(d_U); cuMemFree(d_W); cuMemFree(d_info);
    return (pass_count == fx.B) ? 0 : 1;
}
