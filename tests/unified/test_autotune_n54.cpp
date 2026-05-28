// autotune benchmark at N=54, BPB=1.
//
// Constructs three solvers in sequence:
//   1. solver_suggested    (TuningMode::Suggested) — current behavior (probe).
//   2. solver_autotune     (TuningMode::Autotune)  — runs the sweep.
//   3. solver_autotune2    (TuningMode::Autotune)  — must cache-hit.
//
// For each, runs a 1024-matrix synthetic benchmark with 5-sample medians
// and validates correctness against the existing LAPACK fixture
// (n54_b8_s42_random.bin). Reports a comparative table including the
// autotune sweep duration and the cache-hit flag for the second autotune.

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
#include <random>
#include <string>
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr std::int32_t kMagic    = 0x47454947;
constexpr std::int32_t kVersion  = 1;
constexpr int          kN        = 54;
constexpr int          kBPB      = 1;
constexpr int          kBenchB   = 1024;
constexpr int          kSamples  = 5;
constexpr double       kTolEig   = 1e-10;
constexpr double       kTolVec   = 1e-10;
constexpr double       kDegenGap = 1e-6;
constexpr std::uint64_t kBenchSeed = 0xC0FFEE0042ULL;

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

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct ModeResult {
    std::string label;
    int         block_dim_x   = 0;
    bool        was_autotuned = false;
    bool        sweep_ran     = false;
    double      construct_s   = 0.0;
    double      median_mps    = 0.0;
    double      min_mps       = 0.0;
    double      max_mps       = 0.0;
    double      max_eig_rel   = 0.0;
    double      max_phase     = 0.0;
    bool        correct       = false;
};

ModeResult bench_mode(const std::string& label,
                      TuningMode mode,
                      CUdeviceptr d_H, CUdeviceptr d_S,
                      CUdeviceptr d_U, CUdeviceptr d_W, CUdeviceptr d_info,
                      const Fixture& fx,
                      CUdeviceptr d_Hf, CUdeviceptr d_Sf,
                      CUdeviceptr d_Uf, CUdeviceptr d_Wf, CUdeviceptr d_infof) {
    ModeResult r; r.label = label;
    auto t0 = std::chrono::steady_clock::now();
    UnifiedGeneigSolver solver(kN, /*device_id=*/0, /*bpb=*/kBPB,
                             /*force_block_dim_x=*/0, mode);
    auto t1 = std::chrono::steady_clock::now();
    r.construct_s   = std::chrono::duration<double>(t1 - t0).count();
    r.block_dim_x   = static_cast<int>(solver.block_dim().x);
    r.was_autotuned = solver.was_autotuned();
    r.sweep_ran     = solver.autotune_sweep_ran();

    // Warm-up.
    solver.launch(d_H, d_S, d_W, d_U, d_info, /*batch_size=*/kBPB, nullptr);
    CU_CHECK(cuCtxSynchronize());

    std::vector<double> mps_samples;
    mps_samples.reserve(kSamples);
    for (int s = 0; s < kSamples; ++s) {
        CUevent e0, e1;
        cuEventCreate(&e0, CU_EVENT_DEFAULT);
        cuEventCreate(&e1, CU_EVENT_DEFAULT);
        cuEventRecord(e0, nullptr);
        solver.launch(d_H, d_S, d_W, d_U, d_info, kBenchB, nullptr);
        cuEventRecord(e1, nullptr);
        CU_CHECK(cuCtxSynchronize());
        float ms = 0.f;
        cuEventElapsedTime(&ms, e0, e1);
        cuEventDestroy(e0); cuEventDestroy(e1);
        mps_samples.push_back(static_cast<double>(kBenchB) / (ms / 1000.0));
    }
    r.median_mps = median_of(mps_samples);
    r.min_mps    = *std::min_element(mps_samples.begin(), mps_samples.end());
    r.max_mps    = *std::max_element(mps_samples.begin(), mps_samples.end());

    // Correctness validation against the LAPACK fixture (8 matrices).
    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    solver.launch(d_Hf, d_Sf, d_Wf, d_Uf, d_infof, fx.B, nullptr);
    CU_CHECK(cuCtxSynchronize());
    std::vector<int>     info(fx.B, 0);
    std::vector<cdouble> U_g(mat * fx.B);
    std::vector<double>  W_g(static_cast<std::size_t>(kN) * fx.B);
    CU_CHECK(cuMemcpyDtoH(info.data(), d_infof, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyDtoH(W_g.data(),  d_Wf,    fx.B * kN * sizeof(double)));
    CU_CHECK(cuMemcpyDtoH(U_g.data(),  d_Uf,    mat * fx.B * sizeof(cuDoubleComplex)));
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
    return r;
}

}  // namespace

int main() {
    std::printf("autotune benchmark — N=%d, BPB=%d\n", kN, kBPB);

    // 1024 random matrices for the throughput benchmark.
    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    std::vector<cdouble> H_all(mat * kBenchB);
    std::vector<cdouble> S_all(mat * kBenchB);
    {
        std::mt19937_64 rng(kBenchSeed);
        for (int b = 0; b < kBenchB; ++b) {
            make_hermitian (kN, rng, &H_all[mat * b]);
            make_hpd_random(kN, rng, &S_all[mat * b]);
        }
    }
    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuInit(0));
    CUdevice cu_dev; CUcontext cu_ctx;
    CU_CHECK(cuDeviceGet(&cu_dev, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    CU_CHECK(cuCtxSetCurrent(cu_ctx));

    CU_CHECK(cuMemAlloc(&d_H,    mat * kBenchB * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * kBenchB * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * kBenchB * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    static_cast<std::size_t>(kN) * kBenchB * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, kBenchB * sizeof(int)));
    CU_CHECK(cuMemcpyHtoD(d_H, H_all.data(), mat * kBenchB * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_S, S_all.data(), mat * kBenchB * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemsetD8(d_info, 0, kBenchB * sizeof(int)));

    // Fixture (LAPACK-validated).
    Fixture fx = load_fixture("tests/data/n54_b8_s42_random.bin", kN);
    CUdeviceptr d_Hf = 0, d_Sf = 0, d_Uf = 0, d_Wf = 0, d_infof = 0;
    CU_CHECK(cuMemAlloc(&d_Hf,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_Sf,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_Uf,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_Wf,    static_cast<std::size_t>(kN) * fx.B * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_infof, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyHtoD(d_Hf, fx.H.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_Sf, fx.S.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemsetD8(d_infof, 0, fx.B * sizeof(int)));

    std::vector<ModeResult> results;
    results.push_back(bench_mode("suggested",  TuningMode::Suggested,
                                 d_H, d_S, d_U, d_W, d_info, fx,
                                 d_Hf, d_Sf, d_Uf, d_Wf, d_infof));
    results.push_back(bench_mode("autotune",   TuningMode::Autotune,
                                 d_H, d_S, d_U, d_W, d_info, fx,
                                 d_Hf, d_Sf, d_Uf, d_Wf, d_infof));
    results.push_back(bench_mode("autotune2",  TuningMode::Autotune,
                                 d_H, d_S, d_U, d_W, d_info, fx,
                                 d_Hf, d_Sf, d_Uf, d_Wf, d_infof));

    std::printf("\n");
    std::printf("======================== autotune (N=%d, BPB=%d) ========================\n",
                kN, kBPB);
    std::printf("%-10s | %4s | %10s | %14s | %18s | %15s | %s\n",
                "mode", "BDx", "construct", "median mat/s", "range (min..max)",
                "speedup vs sug", "validation");
    std::printf("-----------+------+------------+----------------+--------------------+-----------------+-----------\n");
    const double base = results.front().median_mps;
    for (const auto& r : results) {
        char range_buf[64];
        std::snprintf(range_buf, sizeof(range_buf),
                      "%.0f .. %.0f", r.min_mps, r.max_mps);
        char spd_buf[32];
        std::snprintf(spd_buf, sizeof(spd_buf), "%.3fx", r.median_mps / base);
        char con_buf[32];
        std::snprintf(con_buf, sizeof(con_buf), "%.1f s", r.construct_s);
        std::printf("%-10s | %4d | %10s | %14.0f | %18s | %15s | %s "
                    "(eig=%.2e, phase=%.2e)%s\n",
                    r.label.c_str(), r.block_dim_x, con_buf,
                    r.median_mps, range_buf, spd_buf,
                    r.correct ? "PASS" : "FAIL", r.max_eig_rel, r.max_phase,
                    r.was_autotuned ? (r.sweep_ran ? "  [sweep]" : "  [cache]") : "");
    }
    std::printf("=====================================================================================\n");

    // Pass criteria.
    bool all_correct = true;
    for (const auto& r : results) all_correct = all_correct && r.correct;
    const ModeResult& sug  = results[0];
    const ModeResult& aut  = results[1];
    const ModeResult& aut2 = results[2];

    bool autotune_not_regression = (aut.median_mps >= 0.95 * sug.median_mps);
    bool cache_hit_speed         = (std::abs(aut2.median_mps - aut.median_mps) <
                                    0.05 * aut.median_mps);
    bool cache_hit_flag          = aut2.was_autotuned && !aut2.sweep_ran;
    bool sweep_ran_first_time    = aut.was_autotuned && aut.sweep_ran;

    std::printf("\nchecks:\n");
    std::printf("  all correct (eig<%.0e && phase<%.0e):              %s\n",
                kTolEig, kTolVec, all_correct ? "PASS" : "FAIL");
    std::printf("  autotune ≥ 0.95 × suggested throughput:           %s "
                "(autotune=%.0f mat/s, suggested=%.0f mat/s)\n",
                autotune_not_regression ? "PASS" : "FAIL",
                aut.median_mps, sug.median_mps);
    std::printf("  autotune2 within 5%% of autotune (cache hit perf): %s "
                "(autotune2=%.0f mat/s, autotune=%.0f mat/s)\n",
                cache_hit_speed ? "PASS" : "FAIL",
                aut2.median_mps, aut.median_mps);
    std::printf("  first autotune ran sweep:                         %s\n",
                sweep_ran_first_time ? "PASS" : "FAIL");
    std::printf("  second autotune hit cache:                        %s\n",
                cache_hit_flag ? "PASS" : "FAIL");

    CU_CHECK(cuMemFree(d_H));    CU_CHECK(cuMemFree(d_S));
    CU_CHECK(cuMemFree(d_U));    CU_CHECK(cuMemFree(d_W));
    CU_CHECK(cuMemFree(d_info));
    CU_CHECK(cuMemFree(d_Hf));   CU_CHECK(cuMemFree(d_Sf));
    CU_CHECK(cuMemFree(d_Uf));   CU_CHECK(cuMemFree(d_Wf));
    CU_CHECK(cuMemFree(d_infof));
    cuDevicePrimaryCtxRelease(cu_dev);

    return (all_correct && autotune_not_regression && cache_hit_speed
            && cache_hit_flag && sweep_ran_first_time) ? 0 : 1;
}
