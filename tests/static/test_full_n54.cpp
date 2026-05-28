// Step 2d test: validate geneig_full_n54_kernel against the LAPACK
// reference fixture (zhegv outputs already stored in fx.W and fx.U).
//
// Two fixtures are exercised: random and near-identity (the latter is closer
// to Sunny's actual physical overlap matrices). Both must pass the same
// criteria.
//
// Comparison metric for U:
//   Generalised eigenvectors satisfy U^H S U = I (S-orthonormal). The right
//   inner product to assess "same eigenvector up to phase" is the S-inner-
//   product:
//       inner = u_gpu^H * (S * u_lapack)
//   For a 1-D eigenspace and unit-S-norm vectors, |inner| = 1 up to fp64
//   epsilon, regardless of the phase factor. Near-degenerate eigenvalues
//   (gap < kDegenGap) are flagged separately.

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

using cdouble = std::complex<double>;

extern "C" void geneig_full_n54_launch(const cuDoubleComplex* d_H,
                                       const cuDoubleComplex* d_S,
                                       double*                d_W,
                                       cuDoubleComplex*       d_U,
                                       int*                   d_info,
                                       cudaStream_t           stream);
extern "C" unsigned int geneig_full_n54_block_dim_x();
extern "C" unsigned int geneig_full_n54_shared_mem_bytes();

namespace {

constexpr std::int32_t kMagic   = 0x47454947;
constexpr std::int32_t kVersion = 1;
constexpr int          kN       = 54;
constexpr double       kTolEig  = 1e-10;
constexpr double       kTolVec  = 1e-10;
constexpr double       kDegenGap = 1e-6;

#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   \
                     #expr, __FILE__, __LINE__, cudaGetErrorString(_e));       \
        std::exit(20);                                                         \
    }                                                                          \
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

// Compute Sv = S * v   (column-major, square N x N).
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
    double max_eig_rel   = 0.0;
    double max_phase_err = 0.0;
};

PerFixtureResult run_fixture(const char* path,
                             cuDoubleComplex* d_H,
                             cuDoubleComplex* d_S,
                             cuDoubleComplex* d_U,
                             double*          d_W,
                             int*             d_info) {
    const Fixture fx = load_fixture(path);
    std::printf("\n--- fixture: %s ---\n", path);
    std::printf("    B=%d, N=%d\n", fx.B, kN);

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    std::vector<cdouble> U_gpu(mat);
    std::vector<double>  W_gpu(kN);
    std::vector<cdouble> Sv(kN);

    PerFixtureResult R;
    for (int b = 0; b < fx.B; ++b) {
        const cdouble* H_b   = &fx.H[mat * b];
        const cdouble* S_b   = &fx.S[mat * b];
        const cdouble* U_ref = &fx.U[mat * b];
        const double*  W_ref = &fx.W[static_cast<std::size_t>(kN) * b];

        // GPU.
        CUDA_CHECK(cudaMemcpy(d_H, H_b, mat * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_S, S_b, mat * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));
        geneig_full_n54_launch(d_H, d_S, d_W, d_U, d_info, /*stream=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());

        int gpu_info = -1;
        CUDA_CHECK(cudaMemcpy(&gpu_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(W_gpu.data(), d_W, kN * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(U_gpu.data(), d_U, mat * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        if (gpu_info != 0) {
            std::printf("    matrix %d: GPU info=%d  FAIL\n", b, gpu_info);
            continue;
        }

        // 1) Eigenvalue comparison vs fixture (LAPACK zhegv outputs).
        double max_eig_rel = 0.0;
        for (int i = 0; i < kN; ++i) {
            const double scale = std::max(std::abs(W_ref[i]), 1.0);
            const double rel   = std::abs(W_gpu[i] - W_ref[i]) / scale;
            max_eig_rel = std::max(max_eig_rel, rel);
        }

        // 2) Generalised eigenvector S-inner-product subspace metric.
        double max_phase_nondegen = 0.0;
        int    degen_excluded     = 0;
        double max_phase_in_degen = 0.0;
        for (int i = 0; i < kN; ++i) {
            const cdouble* u_gpu = &U_gpu[static_cast<std::size_t>(i) * kN];
            const cdouble* u_ref = &U_ref[static_cast<std::size_t>(i) * kN];

            mul_S_v(S_b, u_ref, Sv.data(), kN);            // Sv = S * u_ref

            cdouble inner(0.0, 0.0);
            for (int k = 0; k < kN; ++k) {
                inner += std::conj(u_gpu[k]) * Sv[k];      // u_gpu^H * S * u_ref
            }
            const double phase_err = 1.0 - std::abs(inner);

            const double gap_left  = (i > 0)        ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
            const double gap_right = (i < kN - 1)   ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap < kDegenGap) {
                ++degen_excluded;
                max_phase_in_degen = std::max(max_phase_in_degen, phase_err);
            } else {
                max_phase_nondegen = std::max(max_phase_nondegen, phase_err);
            }
        }

        R.max_eig_rel   = std::max(R.max_eig_rel, max_eig_rel);
        R.max_phase_err = std::max(R.max_phase_err, max_phase_nondegen);

        const bool ok = (max_eig_rel < kTolEig) && (max_phase_nondegen < kTolVec);
        std::printf("    matrix %d: eig_rel=%.3e, phase_err=%.3e",
                    b, max_eig_rel, max_phase_nondegen);
        if (degen_excluded > 0) {
            std::printf(" (degen_excluded=%d, max_phase_in_degen=%.3e)",
                        degen_excluded, max_phase_in_degen);
        }
        std::printf("  %s\n", ok ? "PASS" : "FAIL");
        if (ok) ++R.pass_count;
    }
    std::printf("    SUMMARY: %d/%d passed, max_eig_rel=%.3e, max_phase_err=%.3e\n",
                R.pass_count, fx.B, R.max_eig_rel, R.max_phase_err);
    return R;
}

}  // namespace

int main() {
    std::printf("kernel block_dim.x=%u, total_smem_bytes=%u\n",
                geneig_full_n54_block_dim_x(),
                geneig_full_n54_shared_mem_bytes());

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;

    cuDoubleComplex* d_H = nullptr; cuDoubleComplex* d_S = nullptr;
    cuDoubleComplex* d_U = nullptr; double* d_W = nullptr; int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_H,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_S,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_U,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_W,    static_cast<std::size_t>(kN) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    auto r_random = run_fixture("tests/data/n54_b8_s42_random.bin",
                                d_H, d_S, d_U, d_W, d_info);
    auto r_nearid = run_fixture("tests/data/n54_b8_s42_nearid.bin",
                                d_H, d_S, d_U, d_W, d_info);

    CUDA_CHECK(cudaFree(d_H)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_U)); CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_info));

    const int total_pass = r_random.pass_count + r_nearid.pass_count;
    const int total_run  = 8 + 8;
    std::printf("\nOVERALL: %d/%d passed\n", total_pass, total_run);
    return (total_pass == total_run) ? 0 : 1;
}
