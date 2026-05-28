// multi-N validation of the unified NVRTC kernel.
//
// For each N in {16, 32, 48, 54, 64, 96, 128}:
//   - construct an UnifiedGeneigSolver(N, 0)
//   - load tests/data/n${N}_b8_s42_random.bin   and  ..._nearid.bin
//   - run the unified kernel on each fixture matrix
//   - validate against fixture: max eigenvalue rel-diff < 1e-10,
//                                max phase error < 1e-10
//
// Reports cold compile time per N and prints a summary table at the end.
//
// If a particular N exceeds the GPU's per-block dynamic shared-memory
// ceiling (queried via cuDeviceGetAttribute), or if the constructor throws,
// the test reports SKIPPED for that N and continues. Exit code is 0 only
// if every (N, fixture) pair that was actually run passed.

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

struct Fixture {
    int B = 0; int N = 0;
    std::vector<cdouble> H, S, U;
    std::vector<double>  W;
};

Fixture load_fixture(const std::string& path, int expected_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(10); }
    auto read_pod = [&](auto& v) {
        f.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!f) { std::fprintf(stderr, "header read failed\n"); std::exit(11); }
    };
    std::int32_t magic, version, N, B, seed, reserved;
    read_pod(magic); read_pod(version); read_pod(N);
    read_pod(B);     read_pod(seed);    read_pod(reserved);
    if (magic   != kMagic)   { std::fprintf(stderr, "bad magic\n");     std::exit(12); }
    if (version != kVersion) { std::fprintf(stderr, "bad version\n");   std::exit(13); }
    if (N       != expected_n) {
        std::fprintf(stderr, "fixture %s: expected N=%d got %d\n",
                     path.c_str(), expected_n, N);
        std::exit(14);
    }
    Fixture fx; fx.B = B; fx.N = N;
    const std::size_t mat = static_cast<std::size_t>(N) * N;
    fx.H.resize(mat * B); fx.S.resize(mat * B); fx.U.resize(mat * B);
    fx.W.resize(static_cast<std::size_t>(N) * B);
    auto read_block = [&](void* p, std::size_t bytes) {
        f.read(reinterpret_cast<char*>(p), bytes);
        if (!f) { std::fprintf(stderr, "short read in %s\n", path.c_str()); std::exit(15); }
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

struct PerFixtureResult {
    int    pass_count    = 0;
    int    total         = 0;
    double max_eig_rel   = 0.0;
    double max_phase_err = 0.0;
};

PerFixtureResult run_fixture(UnifiedGeneigSolver& solver,
                             const Fixture& fx,
                             const char* tag) {
    const int n  = fx.N;
    const std::size_t mat = static_cast<std::size_t>(n) * n;

    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    static_cast<std::size_t>(n) * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, sizeof(int)));

    std::vector<cdouble> U_gpu(mat), Sv(n);
    std::vector<double>  W_gpu(n);

    PerFixtureResult R; R.total = fx.B;

    for (int b = 0; b < fx.B; ++b) {
        const cdouble* H_b   = &fx.H[mat * b];
        const cdouble* S_b   = &fx.S[mat * b];
        const cdouble* U_ref = &fx.U[mat * b];
        const double*  W_ref = &fx.W[static_cast<std::size_t>(n) * b];

        CU_CHECK(cuMemcpyHtoD(d_H, H_b, mat * sizeof(cuDoubleComplex)));
        CU_CHECK(cuMemcpyHtoD(d_S, S_b, mat * sizeof(cuDoubleComplex)));
        CU_CHECK(cuMemsetD8(d_info, 0, sizeof(int)));

        solver.launch(d_H, d_S, d_W, d_U, d_info,
                      /*batch_size=*/1, /*stream=*/nullptr);
        CU_CHECK(cuCtxSynchronize());

        int gpu_info = -1;
        CU_CHECK(cuMemcpyDtoH(&gpu_info,    d_info, sizeof(int)));
        CU_CHECK(cuMemcpyDtoH(W_gpu.data(), d_W,    n * sizeof(double)));
        CU_CHECK(cuMemcpyDtoH(U_gpu.data(), d_U,    mat * sizeof(cuDoubleComplex)));
        if (gpu_info != 0) {
            std::printf("    [%s b=%d N=%d] GPU info=%d  FAIL\n", tag, b, n, gpu_info);
            continue;
        }

        double max_eig_rel = 0.0;
        for (int i = 0; i < n; ++i) {
            const double scale = std::max(std::abs(W_ref[i]), 1.0);
            const double rel   = std::abs(W_gpu[i] - W_ref[i]) / scale;
            max_eig_rel = std::max(max_eig_rel, rel);
        }
        double max_phase_nondegen = 0.0;
        int    degen_excluded     = 0;
        for (int i = 0; i < n; ++i) {
            const cdouble* u_gpu = &U_gpu[static_cast<std::size_t>(i) * n];
            const cdouble* u_ref = &U_ref[static_cast<std::size_t>(i) * n];
            mul_S_v(S_b, u_ref, Sv.data(), n);
            cdouble inner(0.0, 0.0);
            for (int k = 0; k < n; ++k) inner += std::conj(u_gpu[k]) * Sv[k];
            const double phase_err = 1.0 - std::abs(inner);
            const double gap_left  = (i > 0)     ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
            const double gap_right = (i < n - 1) ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap < kDegenGap) ++degen_excluded;
            else max_phase_nondegen = std::max(max_phase_nondegen, phase_err);
        }
        R.max_eig_rel   = std::max(R.max_eig_rel,   max_eig_rel);
        R.max_phase_err = std::max(R.max_phase_err, max_phase_nondegen);
        const bool ok = (max_eig_rel < kTolEig) && (max_phase_nondegen < kTolVec);
        if (!ok) {
            std::printf("    [%s b=%d N=%d] eig_rel=%.3e phase=%.3e  FAIL\n",
                        tag, b, n, max_eig_rel, max_phase_nondegen);
        }
        if (ok) ++R.pass_count;
    }

    CU_CHECK(cuMemFree(d_H));
    CU_CHECK(cuMemFree(d_S));
    CU_CHECK(cuMemFree(d_U));
    CU_CHECK(cuMemFree(d_W));
    CU_CHECK(cuMemFree(d_info));
    return R;
}

struct NRow {
    int          n               = 0;
    bool         skipped         = false;
    std::string  skip_reason;
    double       cold_compile_ms = 0.0;
    unsigned     shared_mem      = 0;
    int          bpb             = 0;
    unsigned     block_dim_x     = 0;
    PerFixtureResult random;
    PerFixtureResult nearid;
};

}  // namespace

int main() {
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));

    int max_smem_optin = 0;
    CU_CHECK(cuDeviceGetAttribute(&max_smem_optin,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    std::printf("device 0 max dynamic shared memory per block (optin): %d bytes\n",
                max_smem_optin);

    const std::vector<int> sizes = {16, 32, 48, 54, 64, 96, 128};
    std::vector<NRow> rows;
    rows.reserve(sizes.size());

    for (int n : sizes) {
        NRow row; row.n = n;
        std::printf("\n=== N=%d ===\n", n);

        std::unique_ptr<UnifiedGeneigSolver> solver;
        try {
            auto t0 = std::chrono::steady_clock::now();
            solver = std::make_unique<UnifiedGeneigSolver>(n, /*device_id=*/0);
            auto t1 = std::chrono::steady_clock::now();
            row.cold_compile_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        } catch (const std::exception& e) {
            row.skipped     = true;
            row.skip_reason = std::string("constructor threw: ") + e.what();
            std::printf("  SKIPPED: %s\n", row.skip_reason.c_str());
            rows.push_back(std::move(row));
            continue;
        }

        row.shared_mem  = solver->shared_mem_bytes();
        row.bpb         = solver->batches_per_block();
        row.block_dim_x = solver->block_dim().x;
        std::printf("  cold compile: %.0f ms, BPB=%d, block_dim.x=%u, shared_mem=%u bytes (device max=%d)\n",
                    row.cold_compile_ms, row.bpb, row.block_dim_x,
                    row.shared_mem, max_smem_optin);

        if (static_cast<int>(row.shared_mem) > max_smem_optin) {
            row.skipped     = true;
            row.skip_reason = "shared_mem_required exceeds device max";
            std::printf("  SKIPPED: %s\n", row.skip_reason.c_str());
            rows.push_back(std::move(row));
            continue;
        }

        const std::string base = "tests/data/n" + std::to_string(n) + "_b8_s42_";
        const Fixture fx_random = load_fixture(base + "random.bin", n);
        const Fixture fx_nearid = load_fixture(base + "nearid.bin", n);

        row.random = run_fixture(*solver, fx_random, "random");
        std::printf("  random:  %d/%d pass, max_eig_rel=%.3e, max_phase=%.3e\n",
                    row.random.pass_count, row.random.total,
                    row.random.max_eig_rel, row.random.max_phase_err);

        row.nearid = run_fixture(*solver, fx_nearid, "nearid");
        std::printf("  nearid:  %d/%d pass, max_eig_rel=%.3e, max_phase=%.3e\n",
                    row.nearid.pass_count, row.nearid.total,
                    row.nearid.max_eig_rel, row.nearid.max_phase_err);

        rows.push_back(std::move(row));
    }

    // -----------------------------------------------------------
    // Summary table.
    // -----------------------------------------------------------
    std::printf("\n");
    std::printf("==================== multi-N summary ====================\n");
    std::printf(" %3s | %3s | %5s | %5s | %9s | %9s | %9s | %9s | %s\n",
                "N", "BPB", "BDx", "smem", "cold_ms", "rand_eig", "rand_phs", "near_phs", "status");
    std::printf("-----+-----+------+-------+-----------+-----------+-----------+-----------+--------\n");
    int total_run_pairs = 0, total_pass_pairs = 0;
    for (const auto& r : rows) {
        if (r.skipped) {
            std::printf(" %3d | %3s | %5s | %5s | %9s | %9s | %9s | %9s | SKIP (%s)\n",
                        r.n, "-", "-", "-", "-", "-", "-", "-", r.skip_reason.c_str());
            continue;
        }
        const bool rand_ok = (r.random.pass_count == r.random.total);
        const bool near_ok = (r.nearid.pass_count == r.nearid.total);
        total_run_pairs += 2;
        if (rand_ok) ++total_pass_pairs;
        if (near_ok) ++total_pass_pairs;
        std::printf(" %3d | %3d | %5u | %5u | %9.0f | %9.2e | %9.2e | %9.2e | %s\n",
                    r.n, r.bpb, r.block_dim_x, r.shared_mem, r.cold_compile_ms,
                    r.random.max_eig_rel, r.random.max_phase_err, r.nearid.max_phase_err,
                    (rand_ok && near_ok) ? "PASS" : "FAIL");
    }
    std::printf("==================================================================\n");
    std::printf("OVERALL: %d/%d (N, fixture) pairs passed\n",
                total_pass_pairs, total_run_pairs);

    return (total_run_pairs > 0 && total_pass_pairs == total_run_pairs) ? 0 : 1;
}
