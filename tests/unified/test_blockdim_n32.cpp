// micro-bench: does cuSolverDx's suggested BlockDim<64> at
// (N=32, BPB=2) actually run faster than the previous hardcoded
// BlockDim<128>?  N=32, BPB=2 is the only (N, BPB) tuple in the multi-N
// validation where the suggested value differs from 128.
//
// 1024-matrix throughput run, both block dims, sampled correctness check
// against LAPACK on a few matrices.

#include <cuComplex.h>
#include <cuda.h>
#include <lapacke.h>

#include "unified/unified_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr int    kN        = 32;
constexpr int    kBPB      = 2;
constexpr int    kBatch    = 1024;
constexpr double kTolEig   = 1e-10;
constexpr double kTolVec   = 1e-10;
constexpr double kDegenGap = 1e-6;
constexpr int    kSamplesToCheck = 8;  // matrices to validate against LAPACK

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

void make_hermitian(int n, std::mt19937_64& rng, cdouble* H) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble> A(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[static_cast<std::size_t>(j) * n + i] = cdouble(g(rng), g(rng));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            cdouble aij = A[static_cast<std::size_t>(j) * n + i];
            cdouble aji = A[static_cast<std::size_t>(i) * n + j];
            H[static_cast<std::size_t>(j) * n + i] = aij + std::conj(aji);
        }
}
void make_hpd_random(int n, std::mt19937_64& rng, cdouble* S) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble> A(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[static_cast<std::size_t>(j) * n + i] = cdouble(g(rng), g(rng));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            cdouble acc(0.0, 0.0);
            for (int k = 0; k < n; ++k) {
                cdouble aik = A[static_cast<std::size_t>(k) * n + i];
                cdouble ajk = A[static_cast<std::size_t>(k) * n + j];
                acc += aik * std::conj(ajk);
            }
            S[static_cast<std::size_t>(j) * n + i] = acc;
        }
    for (int i = 0; i < n; ++i)
        S[static_cast<std::size_t>(i) * n + i] += cdouble(static_cast<double>(n), 0.0);
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

struct BenchResult {
    double mat_per_sec  = 0.0;
    double wall_seconds = 0.0;
    int    block_dim_x  = 0;
    bool   correct      = false;
    double max_eig_rel  = 0.0;
    double max_phase    = 0.0;
};

BenchResult run_bench(int force_bdx,
                      const std::vector<cdouble>& H_all,
                      const std::vector<cdouble>& S_all,
                      const std::vector<cdouble>& U_ref,
                      const std::vector<double>&  W_ref,
                      const std::vector<int>&     sample_idx) {
    BenchResult r;
    UnifiedGeneigSolver solver(kN, /*device_id=*/0, /*bpb=*/kBPB,
                             /*force_block_dim_x=*/force_bdx);
    r.block_dim_x = solver.block_dim().x;

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;

    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    static_cast<std::size_t>(kN) * kBatch * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, kBatch * sizeof(int)));

    CU_CHECK(cuMemcpyHtoD(d_H, H_all.data(), mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_S, S_all.data(), mat * kBatch * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemsetD8(d_info, 0, kBatch * sizeof(int)));

    // Warmup.
    solver.launch(d_H, d_S, d_W, d_U, d_info, kBPB, nullptr);
    CU_CHECK(cuCtxSynchronize());

    // Timed full batch.
    CUevent e0, e1;
    cuEventCreate(&e0, CU_EVENT_DEFAULT);
    cuEventCreate(&e1, CU_EVENT_DEFAULT);
    cuEventRecord(e0, nullptr);
    solver.launch(d_H, d_S, d_W, d_U, d_info, kBatch, nullptr);
    cuEventRecord(e1, nullptr);
    CU_CHECK(cuCtxSynchronize());
    float ms = 0.f;
    cuEventElapsedTime(&ms, e0, e1);
    cuEventDestroy(e0); cuEventDestroy(e1);

    r.wall_seconds = ms / 1000.0;
    r.mat_per_sec  = static_cast<double>(kBatch) / r.wall_seconds;

    // Correctness on the sampled subset.
    std::vector<cdouble> U_gpu(mat * kBatch);
    std::vector<double>  W_gpu(static_cast<std::size_t>(kN) * kBatch);
    CU_CHECK(cuMemcpyDtoH(W_gpu.data(), d_W, kBatch * kN * sizeof(double)));
    CU_CHECK(cuMemcpyDtoH(U_gpu.data(), d_U, mat * kBatch * sizeof(cuDoubleComplex)));

    std::vector<cdouble> Sv(kN);
    int                  bad_pairs = 0;
    for (int b : sample_idx) {
        const cdouble* S_b = &S_all[mat * b];
        const cdouble* U_r = &U_ref[mat * b];
        const double*  W_r = &W_ref[static_cast<std::size_t>(kN) * b];
        const cdouble* U_g = &U_gpu[mat * b];
        const double*  W_g = &W_gpu[static_cast<std::size_t>(kN) * b];

        double m_eig = 0.0;
        for (int i = 0; i < kN; ++i) {
            const double scale = std::max(std::abs(W_r[i]), 1.0);
            m_eig = std::max(m_eig, std::abs(W_g[i] - W_r[i]) / scale);
        }
        double m_phase = 0.0;
        for (int i = 0; i < kN; ++i) {
            const cdouble* u_g = &U_g[static_cast<std::size_t>(i) * kN];
            const cdouble* u_r = &U_r[static_cast<std::size_t>(i) * kN];
            mul_S_v(S_b, u_r, Sv.data(), kN);
            cdouble inner(0.0, 0.0);
            for (int k = 0; k < kN; ++k) inner += std::conj(u_g[k]) * Sv[k];
            const double phase_err = 1.0 - std::abs(inner);
            const double gap_left  = (i > 0)     ? std::abs(W_r[i] - W_r[i - 1]) : INFINITY;
            const double gap_right = (i < kN - 1) ? std::abs(W_r[i + 1] - W_r[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap >= kDegenGap) m_phase = std::max(m_phase, phase_err);
        }
        r.max_eig_rel = std::max(r.max_eig_rel, m_eig);
        r.max_phase   = std::max(r.max_phase,   m_phase);
        if (m_eig >= kTolEig || m_phase >= kTolVec) ++bad_pairs;
    }
    r.correct = (bad_pairs == 0);

    CU_CHECK(cuMemFree(d_H)); CU_CHECK(cuMemFree(d_S));
    CU_CHECK(cuMemFree(d_U)); CU_CHECK(cuMemFree(d_W));
    CU_CHECK(cuMemFree(d_info));
    return r;
}

}  // namespace

int main() {
    // ---- Generate workload ----
    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    std::vector<cdouble> H_all(mat * kBatch);
    std::vector<cdouble> S_all(mat * kBatch);
    std::mt19937_64 rng(0xBEEFC0DE32ULL);
    for (int b = 0; b < kBatch; ++b) {
        make_hermitian(kN, rng, &H_all[mat * b]);
        make_hpd_random(kN, rng, &S_all[mat * b]);
    }

    // LAPACK reference on a sampled subset only — full 1024-matrix
    // host-side reference would dominate runtime and is unnecessary for
    // a throughput micro-bench.
    std::vector<int> sample_idx;
    for (int i = 0; i < kSamplesToCheck; ++i) {
        sample_idx.push_back((i * kBatch) / kSamplesToCheck);
    }
    std::vector<cdouble> U_ref(mat * kBatch);
    std::vector<double>  W_ref(static_cast<std::size_t>(kN) * kBatch);
    for (int b : sample_idx) {
        std::vector<cdouble> H_w(mat), S_w(mat);
        std::memcpy(H_w.data(), &H_all[mat * b], sizeof(cdouble) * mat);
        std::memcpy(S_w.data(), &S_all[mat * b], sizeof(cdouble) * mat);
        lapack_int li = LAPACKE_zhegv(
            LAPACK_COL_MAJOR, /*itype=*/1, 'V', 'L', kN,
            reinterpret_cast<lapack_complex_double*>(H_w.data()), kN,
            reinterpret_cast<lapack_complex_double*>(S_w.data()), kN,
            &W_ref[static_cast<std::size_t>(kN) * b]);
        if (li != 0) { std::fprintf(stderr, "zhegv failed at b=%d, info=%d\n", b, (int)li); return 1; }
        std::memcpy(&U_ref[mat * b], H_w.data(), sizeof(cdouble) * mat);
    }
    std::printf("loaded %d sampled LAPACK references\n", (int)sample_idx.size());

    // ---- Run both block-dim variants ----
    std::printf("\n=== suggested BlockDim (probe) ===\n");
    BenchResult r_suggested = run_bench(/*force_bdx=*/0,  // 0 = use probe
                                        H_all, S_all, U_ref, W_ref, sample_idx);
    std::printf("BDx=%d, %.0f mat/sec, wall=%.4f s, max_eig_rel=%.3e, max_phase=%.3e, %s\n",
                r_suggested.block_dim_x, r_suggested.mat_per_sec, r_suggested.wall_seconds,
                r_suggested.max_eig_rel, r_suggested.max_phase,
                r_suggested.correct ? "PASS" : "FAIL");

    std::printf("\n=== forced BlockDim<128> (3.5a-equivalent) ===\n");
    BenchResult r_forced = run_bench(/*force_bdx=*/128,
                                     H_all, S_all, U_ref, W_ref, sample_idx);
    std::printf("BDx=%d, %.0f mat/sec, wall=%.4f s, max_eig_rel=%.3e, max_phase=%.3e, %s\n",
                r_forced.block_dim_x, r_forced.mat_per_sec, r_forced.wall_seconds,
                r_forced.max_eig_rel, r_forced.max_phase,
                r_forced.correct ? "PASS" : "FAIL");

    const double ratio = r_suggested.mat_per_sec / r_forced.mat_per_sec;
    std::printf("\nN=%d, BPB=%d, B=%d:\n", kN, kBPB, kBatch);
    std::printf("  BlockDim<%d>  (suggested): %.0f mat/sec\n",
                r_suggested.block_dim_x, r_suggested.mat_per_sec);
    std::printf("  BlockDim<%d> (forced):    %.0f mat/sec\n",
                r_forced.block_dim_x, r_forced.mat_per_sec);
    std::printf("  ratio (suggested / forced): %.3f\n", ratio);

    if (!r_suggested.correct || !r_forced.correct) {
        std::printf("FAIL: correctness regression detected\n");
        return 1;
    }
    return 0;
}
