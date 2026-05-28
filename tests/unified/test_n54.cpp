// regression test — unified NVRTC pipeline at N=54 against the
// random-S fixture. Kept as a fast single-N smoke check; the broader multi-N
// validation lives in test_nvrtc_multiN.cpp.
//
// The hybrid two-kernel split that used is gone : with the
// cuSolverDx header overlay (src/common/cusolverdx_overlay/) the heev path
// instantiates under NVRTC and the kernel runs all five stages in one
// launch. See docs/CUSOLVERDX_NVRTC_HEEV_BUG.md.

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
        if (!f) { std::fprintf(stderr, "header read failed\n"); std::exit(11); }
    };
    std::int32_t magic, version, N, B, seed, reserved;
    read_pod(magic); read_pod(version); read_pod(N);
    read_pod(B);     read_pod(seed);    read_pod(reserved);
    if (magic   != kMagic)   { std::fprintf(stderr, "bad magic 0x%08x\n", magic); std::exit(12); }
    if (version != kVersion) { std::fprintf(stderr, "bad version %d\n", version); std::exit(13); }
    if (N       != kN)       { std::fprintf(stderr, "expected N=%d got %d\n", kN, N); std::exit(14); }
    Fixture fx; fx.B = B;
    const std::size_t mat = static_cast<std::size_t>(N) * N;
    fx.H.resize(mat * B); fx.S.resize(mat * B); fx.U.resize(mat * B);
    fx.W.resize(static_cast<std::size_t>(N) * B);
    auto read_block = [&](void* p, std::size_t bytes) {
        f.read(reinterpret_cast<char*>(p), bytes);
        if (!f) { std::fprintf(stderr, "short read\n"); std::exit(15); }
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

}  // namespace

int main() {
    auto t0 = std::chrono::steady_clock::now();
    UnifiedGeneigSolver solver(kN, /*device_id=*/0);
    auto t1 = std::chrono::steady_clock::now();
    const double cold_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("cold NVRTC compile: %.1f ms\n", cold_ms);
    std::printf("kernel block_dim.x=%u, shared_mem_bytes=%u, matrix_size=%d\n",
                solver.block_dim().x, solver.shared_mem_bytes(), solver.matrix_size());

    const Fixture fx = load_fixture("tests/data/n54_b8_s42_random.bin");
    std::printf("loaded fixture: B=%d, N=%d\n", fx.B, kN);

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;

    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    static_cast<std::size_t>(kN) * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, sizeof(int)));

    std::vector<cdouble> U_gpu(mat), Sv(kN);
    std::vector<double>  W_gpu(kN);

    int    pass_count = 0;
    double max_eig_rel_overall = 0.0;
    double max_phase_overall   = 0.0;

    for (int b = 0; b < fx.B; ++b) {
        const cdouble* H_b   = &fx.H[mat * b];
        const cdouble* S_b   = &fx.S[mat * b];
        const cdouble* U_ref = &fx.U[mat * b];
        const double*  W_ref = &fx.W[static_cast<std::size_t>(kN) * b];

        CU_CHECK(cuMemcpyHtoD(d_H, H_b, mat * sizeof(cuDoubleComplex)));
        CU_CHECK(cuMemcpyHtoD(d_S, S_b, mat * sizeof(cuDoubleComplex)));
        CU_CHECK(cuMemsetD8(d_info, 0, sizeof(int)));

        solver.launch(d_H, d_S, d_W, d_U, d_info,
                      /*batch_size=*/1, /*stream=*/nullptr);
        CU_CHECK(cuCtxSynchronize());

        int gpu_info = -1;
        CU_CHECK(cuMemcpyDtoH(&gpu_info,    d_info, sizeof(int)));
        CU_CHECK(cuMemcpyDtoH(W_gpu.data(), d_W,    kN * sizeof(double)));
        CU_CHECK(cuMemcpyDtoH(U_gpu.data(), d_U,    mat * sizeof(cuDoubleComplex)));
        if (gpu_info != 0) {
            std::printf("matrix %d: GPU info=%d  FAIL\n", b, gpu_info);
            continue;
        }

        double max_eig_rel = 0.0;
        for (int i = 0; i < kN; ++i) {
            const double scale = std::max(std::abs(W_ref[i]), 1.0);
            const double rel   = std::abs(W_gpu[i] - W_ref[i]) / scale;
            max_eig_rel = std::max(max_eig_rel, rel);
        }
        double max_phase_nondegen = 0.0;
        int    degen_excluded     = 0;
        for (int i = 0; i < kN; ++i) {
            const cdouble* u_gpu = &U_gpu[static_cast<std::size_t>(i) * kN];
            const cdouble* u_ref = &U_ref[static_cast<std::size_t>(i) * kN];
            mul_S_v(S_b, u_ref, Sv.data(), kN);
            cdouble inner(0.0, 0.0);
            for (int k = 0; k < kN; ++k) inner += std::conj(u_gpu[k]) * Sv[k];
            const double phase_err = 1.0 - std::abs(inner);
            const double gap_left  = (i > 0)        ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
            const double gap_right = (i < kN - 1)   ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap < kDegenGap) ++degen_excluded;
            else max_phase_nondegen = std::max(max_phase_nondegen, phase_err);
        }
        max_eig_rel_overall = std::max(max_eig_rel_overall, max_eig_rel);
        max_phase_overall   = std::max(max_phase_overall, max_phase_nondegen);
        const bool ok = (max_eig_rel < kTolEig) && (max_phase_nondegen < kTolVec);
        std::printf("matrix %d: eig_rel=%.3e, phase_err=%.3e", b, max_eig_rel, max_phase_nondegen);
        if (degen_excluded > 0) std::printf(" (degen_excluded=%d)", degen_excluded);
        std::printf("  %s\n", ok ? "PASS" : "FAIL");
        if (ok) ++pass_count;
    }
    std::printf("SUMMARY: %d/%d passed, max_eig_rel=%.3e, max_phase=%.3e\n",
                pass_count, fx.B, max_eig_rel_overall, max_phase_overall);

    CU_CHECK(cuMemFree(d_H));
    CU_CHECK(cuMemFree(d_S));
    CU_CHECK(cuMemFree(d_U));
    CU_CHECK(cuMemFree(d_W));
    CU_CHECK(cuMemFree(d_info));

    return (pass_count == fx.B) ? 0 : 1;
}
