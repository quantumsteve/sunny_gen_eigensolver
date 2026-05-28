// exercise the C ABI shim end-to-end before any Julia
// involvement. Loads the n54_b8_s42_random fixture, drives the solver
// through sunny_geneig_create / _launch / _destroy, validates against
// LAPACK reference results stored in the fixture, and verifies all
// introspection accessors return sane values.

#include "common/c_api.h"
#include <cuComplex.h>
#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

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

}  // namespace

int main() {
    std::printf("C ABI test — exercising libsunny_geneig.so via dlopen-equivalent linkage\n");

    // Step 1: NULL-handle defensive checks.
    if (sunny_geneig_matrix_size(nullptr) != -1) {
        std::fprintf(stderr, "matrix_size(NULL) should return -1\n");
        return 30;
    }
    if (sunny_geneig_batches_per_block(nullptr) != -1) {
        std::fprintf(stderr, "batches_per_block(NULL) should return -1\n");
        return 31;
    }
    if (sunny_geneig_block_dim_x(nullptr) != -1) {
        std::fprintf(stderr, "block_dim_x(NULL) should return -1\n");
        return 32;
    }
    if (sunny_geneig_shared_mem_bytes(nullptr) != 0u) {
        std::fprintf(stderr, "shared_mem_bytes(NULL) should return 0\n");
        return 33;
    }
    sunny_geneig_destroy(nullptr);  // must be safe with NULL
    std::printf("  NULL-handle accessors: OK\n");

    // Step 2: launch with NULL solver should report INVALID_ARG, not crash.
    auto bad = sunny_geneig_launch(nullptr, 0, 0, 0, 0, 0, 8, 0);
    if (bad != SUNNY_GENEIG_INVALID_ARG) {
        std::fprintf(stderr, "launch(NULL, ...) returned %d, expected INVALID_ARG\n", bad);
        return 34;
    }
    const char* err = sunny_geneig_last_error();
    if (!err || std::strlen(err) == 0) {
        std::fprintf(stderr, "last_error after NULL launch should be non-empty\n");
        return 35;
    }
    std::printf("  NULL-solver launch returns INVALID_ARG with msg: \"%s\"\n", err);

    // Step 3: real construction. Suggested mode + Auto cache so repeated
    // runs of test_c_api are fast (disk-hit from prior phases).
    sunny_geneig_solver_t s = sunny_geneig_create(
        kN, /*device_id=*/0, /*bpb_request=*/kBPB,
        SUNNY_GENEIG_TUNING_SUGGESTED, SUNNY_GENEIG_CACHE_AUTO);
    if (!s) {
        std::fprintf(stderr, "sunny_geneig_create failed: %s\n",
                     sunny_geneig_last_error());
        return 40;
    }
    std::printf("  create(N=%d, bpb=%d): OK\n", kN, kBPB);
    std::printf("  matrix_size       = %d\n",  sunny_geneig_matrix_size(s));
    std::printf("  batches_per_block = %d\n",  sunny_geneig_batches_per_block(s));
    std::printf("  block_dim_x       = %d\n",  sunny_geneig_block_dim_x(s));
    std::printf("  shared_mem_bytes  = %u\n",  sunny_geneig_shared_mem_bytes(s));

    if (sunny_geneig_matrix_size(s) != kN
        || sunny_geneig_batches_per_block(s) <= 0
        || sunny_geneig_block_dim_x(s) <= 0
        || sunny_geneig_shared_mem_bytes(s) == 0u) {
        std::fprintf(stderr, "introspection returned bad values\n");
        sunny_geneig_destroy(s);
        return 41;
    }

    // Step 4: load fixture, allocate device buffers, run launch.
    Fixture fx = load_fixture("tests/data/n54_b8_s42_random.bin", kN);
    const std::size_t mat = static_cast<std::size_t>(kN) * kN;

    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuInit(0));
    CUdevice cu_dev; CUcontext cu_ctx;
    CU_CHECK(cuDeviceGet(&cu_dev, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    CU_CHECK(cuCtxSetCurrent(cu_ctx));
    CU_CHECK(cuMemAlloc(&d_H,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_S,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_U,    mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemAlloc(&d_W,    static_cast<std::size_t>(kN) * fx.B * sizeof(double)));
    CU_CHECK(cuMemAlloc(&d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyHtoD(d_H, fx.H.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemcpyHtoD(d_S, fx.S.data(), mat * fx.B * sizeof(cuDoubleComplex)));
    CU_CHECK(cuMemsetD8(d_info, 0, fx.B * sizeof(int)));

    auto status = sunny_geneig_launch(
        s,
        static_cast<uintptr_t>(d_H), static_cast<uintptr_t>(d_S),
        static_cast<uintptr_t>(d_W), static_cast<uintptr_t>(d_U),
        static_cast<uintptr_t>(d_info),
        fx.B, /*stream=*/0);
    if (status != SUNNY_GENEIG_OK) {
        std::fprintf(stderr, "launch failed with status=%d: %s\n",
                     status, sunny_geneig_last_error());
        sunny_geneig_destroy(s);
        return 42;
    }
    CU_CHECK(cuCtxSynchronize());

    // Step 5: validate.
    std::vector<int>     info(fx.B, 0);
    std::vector<cdouble> U_g(mat * fx.B);
    std::vector<double>  W_g(static_cast<std::size_t>(kN) * fx.B);
    CU_CHECK(cuMemcpyDtoH(info.data(), d_info, fx.B * sizeof(int)));
    CU_CHECK(cuMemcpyDtoH(W_g.data(),  d_W,    fx.B * kN * sizeof(double)));
    CU_CHECK(cuMemcpyDtoH(U_g.data(),  d_U,    mat * fx.B * sizeof(cuDoubleComplex)));

    int    pass = 0;
    double max_eig_rel = 0.0, max_phase = 0.0;
    std::vector<cdouble> Sv(kN);
    for (int b = 0; b < fx.B; ++b) {
        if (info[b] != 0) continue;
        const cdouble* S_b   = &fx.S[mat * b];
        const cdouble* U_ref = &fx.U[mat * b];
        const double*  W_ref = &fx.W[static_cast<std::size_t>(kN) * b];
        double m_eig = 0.0, m_phase = 0.0;
        for (int i = 0; i < kN; ++i) {
            const double scale = std::max(std::abs(W_ref[i]), 1.0);
            m_eig = std::max(m_eig,
                std::abs(W_g[static_cast<std::size_t>(kN) * b + i] - W_ref[i]) / scale);
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
            if (gap >= kDegenGap) m_phase = std::max(m_phase, phase_err);
        }
        max_eig_rel = std::max(max_eig_rel, m_eig);
        max_phase   = std::max(max_phase,   m_phase);
        if (m_eig < kTolEig && m_phase < kTolVec) ++pass;
    }
    std::printf("  validation: %d/%d PASS (max_eig_rel=%.3e, max_phase=%.3e)\n",
                pass, fx.B, max_eig_rel, max_phase);

    // Step 6: clean up.
    cuMemFree(d_H); cuMemFree(d_S); cuMemFree(d_U); cuMemFree(d_W); cuMemFree(d_info);
    sunny_geneig_destroy(s);
    cuDevicePrimaryCtxRelease(cu_dev);

    return (pass == fx.B && max_eig_rel < kTolEig && max_phase < kTolVec) ? 0 : 1;
}
