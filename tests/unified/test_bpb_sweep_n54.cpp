// BatchesPerBlock sweep at N=54.
//
// For each BPB ∈ {1, 2, 4, 8} attempt to construct a solver with explicit BPB
// override; on success, run the n54_b32_s123_random fixture (32 matrices,
// divisible by every BPB candidate) and validate at fp64 epsilon.
// On the development hardware (RTX 5070 Ti laptop, 101 KB per-block), only
// BPB=1 is expected to fit; BPB ≥ 2 should fail construction with the
// informative "exceeds device budget" message. On H100 (228 KB), BPB=2
// should fit (~199 KB at N=54); BPB=4+ won't.

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
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <string>
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

struct SweepRow {
    int          bpb            = 0;
    bool         constructed    = false;
    std::string  reject_reason;
    double       throughput_mps = 0.0;
    double       max_eig_rel    = 0.0;
    double       max_phase      = 0.0;
    int          pass_count     = 0;
    int          total          = 0;
    unsigned     block_dim_x    = 0;
};

SweepRow run_one(int bpb, const Fixture& fx) {
    SweepRow r; r.bpb = bpb; r.total = fx.B;

    std::unique_ptr<UnifiedGeneigSolver> solver;
    try {
        solver = std::make_unique<UnifiedGeneigSolver>(kN, /*device_id=*/0, bpb);
    } catch (const std::exception& e) {
        r.reject_reason = e.what();
        return r;
    }
    r.constructed = true;
    r.block_dim_x = solver->block_dim().x;

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    static_cast<std::size_t>(kN) * fx.B * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, fx.B * sizeof(int)));

    CU_CHECK(cuMemcpyHtoD(d_H, fx.H.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_S, fx.S.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemsetD8(d_info, 0, fx.B * sizeof(int)));

    // Warm-up launch (one batch worth) to pre-resolve any first-launch overhead.
    solver->launch(d_H, d_S, d_W, d_U, d_info, bpb, /*stream=*/nullptr);
    CU_CHECK(cuCtxSynchronize());

    // Timed run on the full fixture.
    CUevent e0, e1;
    cuEventCreate(&e0, CU_EVENT_DEFAULT);
    cuEventCreate(&e1, CU_EVENT_DEFAULT);
    cuEventRecord(e0, /*stream=*/nullptr);
    solver->launch(d_H, d_S, d_W, d_U, d_info, fx.B, /*stream=*/nullptr);
    cuEventRecord(e1, /*stream=*/nullptr);
    CU_CHECK(cuCtxSynchronize());
    float ms = 0.f;
    cuEventElapsedTime(&ms, e0, e1);
    cuEventDestroy(e0); cuEventDestroy(e1);
    r.throughput_mps = (double)fx.B / (ms / 1000.0);

    std::vector<int>     info(fx.B, 0);
    std::vector<cdouble> U_gpu(mat * fx.B);
    std::vector<double>  W_gpu(static_cast<std::size_t>(kN) * fx.B);
    CU_CHECK(cuMemcpyDtoH(info.data(),  d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyDtoH(W_gpu.data(), d_W,    fx.B * kN * sizeof(double)));
    CU_CHECK(cuMemcpyDtoH(U_gpu.data(), d_U,    mat * fx.B * sizeof(cuDoubleComplex)));

    std::vector<cdouble> Sv(kN);
    for (int b = 0; b < fx.B; ++b) {
        if (info[b] != 0) continue;
        const cdouble* S_b   = &fx.S[mat * b];
        const cdouble* U_ref = &fx.U[mat * b];
        const double*  W_ref = &fx.W[static_cast<std::size_t>(kN) * b];
        const cdouble* U_g   = &U_gpu[mat * b];
        const double*  W_g   = &W_gpu[static_cast<std::size_t>(kN) * b];

        double m_eig = 0.0;
        for (int i = 0; i < kN; ++i) {
            const double scale = std::max(std::abs(W_ref[i]), 1.0);
            m_eig = std::max(m_eig, std::abs(W_g[i] - W_ref[i]) / scale);
        }
        double m_phase = 0.0;
        for (int i = 0; i < kN; ++i) {
            const cdouble* u_g = &U_g[static_cast<std::size_t>(i) * kN];
            const cdouble* u_r = &U_ref[static_cast<std::size_t>(i) * kN];
            mul_S_v(S_b, u_r, Sv.data(), kN);
            cdouble inner(0.0, 0.0);
            for (int k = 0; k < kN; ++k) inner += std::conj(u_g[k]) * Sv[k];
            const double phase_err = 1.0 - std::abs(inner);
            const double gap_left  = (i > 0)     ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
            const double gap_right = (i < kN - 1) ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap >= kDegenGap) m_phase = std::max(m_phase, phase_err);
        }
        r.max_eig_rel = std::max(r.max_eig_rel, m_eig);
        r.max_phase   = std::max(r.max_phase,   m_phase);
        if (m_eig < kTolEig && m_phase < kTolVec) ++r.pass_count;
    }

    CU_CHECK(cuMemFree(d_H));
    CU_CHECK(cuMemFree(d_S));
    CU_CHECK(cuMemFree(d_U));
    CU_CHECK(cuMemFree(d_W));
    CU_CHECK(cuMemFree(d_info));
    return r;
}

// Synthetic 1024-matrix generator matching test_full_batched_n54.cpp's
// test_large_batch. Same seed and same construction so the workload is
// equivalent across the two test paths.
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

// Throughput benchmark: 1024 synthetic matrices at N=54, BPB=1, NVRTC path.
// Mirrors the workload in tests/static/test_full_batched_n54.cpp's test_large_batch
// (same seed, same construction). The number to compare against the static-
// link path's baseline.
double run_throughput_benchmark_1024_bpb1() {
    constexpr int kBatch = 1024;
    constexpr std::uint64_t kSeed = 0xC0FFEE0042ULL;

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    std::vector<cdouble> H_all(mat * kBatch);
    std::vector<cdouble> S_all(mat * kBatch);
    std::mt19937_64 rng(kSeed);
    for (int b = 0; b < kBatch; ++b) {
        make_hermitian(kN, rng, &H_all[mat * b]);
        make_hpd_random(kN, rng, &S_all[mat * b]);
    }

    UnifiedGeneigSolver solver(kN, /*device_id=*/0, /*bpb=*/1);

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
    solver.launch(d_H, d_S, d_W, d_U, d_info, /*batch_size=*/1, nullptr);
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

    CU_CHECK(cuMemFree(d_H)); CU_CHECK(cuMemFree(d_S));
    CU_CHECK(cuMemFree(d_U)); CU_CHECK(cuMemFree(d_W));
    CU_CHECK(cuMemFree(d_info));

    const double secs = ms / 1000.0;
    const double mps  = static_cast<double>(kBatch) / secs;
    std::printf("\n--- 1024-matrix NVRTC throughput benchmark (BPB=1) ---\n");
    std::printf("  wall: %.3f s; %.0f matrices/sec\n", secs, mps);
    return mps;
}

}  // namespace

int main() {
    const Fixture fx = load_fixture("tests/data/n54_b32_s123_random.bin", kN);
    std::printf("loaded fixture: B=%d, N=%d\n", fx.B, kN);

    const std::vector<int> bpbs = {1, 2, 4, 8};
    std::vector<SweepRow> rows;
    for (int bpb : bpbs) {
        std::printf("\n--- BPB=%d ---\n", bpb);
        SweepRow r = run_one(bpb, fx);
        if (!r.constructed) {
            std::printf("  SKIP: %s\n", r.reject_reason.c_str());
        } else {
            std::printf("  PASS=%d/%d, max_eig_rel=%.3e, max_phase=%.3e, throughput=%.0f mat/s\n",
                        r.pass_count, r.total, r.max_eig_rel, r.max_phase, r.throughput_mps);
        }
        rows.push_back(std::move(r));
    }

    std::printf("\n");
    std::printf("==================== BPB sweep summary (N=54, B=%d) ====================\n", fx.B);
    std::printf(" %3s | %3s | %13s | %12s | %11s | %11s | %s\n",
                "BPB", "BDx", "constructable", "matrices/sec", "max_eig_rel", "max_phase", "status");
    std::printf("-----+-----+---------------+--------------+-------------+-------------+--------\n");
    int passed_count = 0, attempted_count = 0;
    for (const auto& r : rows) {
        if (!r.constructed) {
            std::printf(" %3d | %3s | %13s | %12s | %11s | %11s | SKIP (%s)\n",
                        r.bpb, "-", "no", "-", "-", "-",
                        r.reject_reason.substr(0, 60).c_str());
            continue;
        }
        ++attempted_count;
        const bool ok = (r.pass_count == r.total) && (r.max_eig_rel < kTolEig) && (r.max_phase < kTolVec);
        if (ok) ++passed_count;
        std::printf(" %3d | %3u | %13s | %12.0f | %11.3e | %11.3e | %s\n",
                    r.bpb, r.block_dim_x, "yes", r.throughput_mps, r.max_eig_rel, r.max_phase,
                    ok ? "PASS" : "FAIL");
    }
    std::printf("=========================================================================\n");
    std::printf("OVERALL: %d/%d constructable BPB values passed correctness\n",
                passed_count, attempted_count);

    // Throughput benchmark for direct comparison against the
    // static-link baseline (tests/static/test_full_batched_n54's test_large_batch).
    const double mps_1024 = run_throughput_benchmark_1024_bpb1();
    std::printf("\nNVRTC throughput at N=54, B=1024, BPB=1: %.0f mat/s\n", mps_1024);
    std::printf("(Compare to static path baseline reported by\n");
    std::printf(" tests/static/test_full_batched_n54's synthetic 1024-matrix test.)\n");

    return (attempted_count > 0 && passed_count == attempted_count) ? 0 : 1;
}
