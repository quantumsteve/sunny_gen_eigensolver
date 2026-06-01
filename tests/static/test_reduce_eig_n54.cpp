// Step 2c test: validate geneig_reduce_eig_n54_kernel against
// LAPACKE_zpotrf + LAPACKE_zhegst + LAPACKE_zheev (the QR-algorithm variant
// matching cuSolverDx's heev).
//
// Loads tests/data/n54_b8_s42_random.bin (8 generic 54x54 H/S pairs), runs the
// GPU reduce+eig kernel, computes the LAPACK reference path, and compares:
//   1. Eigenvalues element-wise (max relative diff, normalised by max(|λ|, 1)).
//   2. Eigenvectors column-by-column via subspace metric: for each column i,
//      phase_error_i = 1 - |⟨V_gpu[:,i], V_lapack[:,i]⟩|. With unit-norm
//      eigenvectors spanning the same subspace this is ~ fp64 epsilon. Near
//      degeneracies (eigenvalue gap < 1e-6) make this quantity unreliable —
//      we flag affected indices but don't fail on them.
//
// Pass criteria: max eigenvalue rel_diff < 1e-10, max phase_error < 1e-10.

#include <cuComplex.h>
#include <cuda_runtime.h>
#include "mkl.h"

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

extern "C" void geneig_reduce_eig_n54_launch(const cuDoubleComplex* d_H,
                                             const cuDoubleComplex* d_S,
                                             double*                d_W,
                                             cuDoubleComplex*       d_V,
                                             int*                   d_info,
                                             cudaStream_t           stream);
extern "C" unsigned int geneig_reduce_eig_n54_block_dim_x();
extern "C" unsigned int geneig_reduce_eig_n54_shared_mem_bytes();
extern "C" unsigned int geneig_reduce_eig_n54_chol_smem_bytes();
extern "C" unsigned int geneig_reduce_eig_n54_heev_smem_bytes();
extern "C" unsigned int geneig_reduce_eig_n54_heev_workspace();

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

}  // namespace

int main() {
    const Fixture fx = load_fixture("tests/data/n54_b8_s42_random.bin");
    std::printf("loaded fixture: B=%d, N=%d\n", fx.B, kN);
    std::printf("kernel block_dim.x=%u, total_smem_bytes=%u\n",
                geneig_reduce_eig_n54_block_dim_x(),
                geneig_reduce_eig_n54_shared_mem_bytes());
    std::printf("  Cholesky54::shared_memory_size = %u bytes\n",
                geneig_reduce_eig_n54_chol_smem_bytes());
    std::printf("  Heev54::shared_memory_size     = %u bytes\n",
                geneig_reduce_eig_n54_heev_smem_bytes());
    std::printf("  Heev54::workspace_size         = %u (DataType elements)\n",
                geneig_reduce_eig_n54_heev_workspace());

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;

    cuDoubleComplex* d_H = nullptr; cuDoubleComplex* d_S = nullptr;
    cuDoubleComplex* d_V = nullptr; double* d_W = nullptr; int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_H,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_S,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_V,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_W,    static_cast<std::size_t>(kN) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    std::vector<cdouble> V_gpu(mat), V_lapack(mat), H_lapack(mat), S_lapack(mat);
    std::vector<double>  W_gpu(kN), W_lapack(kN);

    int    pass_count = 0;
    double max_eig_rel_overall = 0.0;
    double max_phase_overall   = 0.0;

    for (int b = 0; b < fx.B; ++b) {
        const cdouble* H_b = &fx.H[mat * b];
        const cdouble* S_b = &fx.S[mat * b];

        // GPU path.
        CUDA_CHECK(cudaMemcpy(d_H, H_b, mat * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_S, S_b, mat * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));
        geneig_reduce_eig_n54_launch(d_H, d_S, d_W, d_V, d_info, /*stream=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());

        int gpu_info = -1;
        CUDA_CHECK(cudaMemcpy(&gpu_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(W_gpu.data(), d_W, kN * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(V_gpu.data(), d_V, mat * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        if (gpu_info != 0) {
            std::printf("matrix %d: GPU info=%d  FAIL\n", b, gpu_info);
            continue;
        }

        // LAPACK reference: zpotrf → zhegst → zheev.
        std::memcpy(S_lapack.data(), S_b, mat * sizeof(cdouble));
        std::memcpy(H_lapack.data(), H_b, mat * sizeof(cdouble));
        lapack_int li;
        li = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'L', kN,
            reinterpret_cast<lapack_complex_double*>(S_lapack.data()), kN);
        if (li != 0) { std::printf("matrix %d: zpotrf info=%d  FAIL\n", b, (int)li); continue; }
        li = LAPACKE_zhegst(LAPACK_COL_MAJOR, /*itype=*/1, 'L', kN,
            reinterpret_cast<lapack_complex_double*>(H_lapack.data()), kN,
            reinterpret_cast<lapack_complex_double*>(S_lapack.data()), kN);
        if (li != 0) { std::printf("matrix %d: zhegst info=%d  FAIL\n", b, (int)li); continue; }
        li = LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'L', kN,
            reinterpret_cast<lapack_complex_double*>(H_lapack.data()), kN,
            W_lapack.data());
        if (li != 0) { std::printf("matrix %d: zheev info=%d  FAIL\n", b, (int)li); continue; }
        // After zheev with jobz='V', H_lapack columns are the eigenvectors.
        std::memcpy(V_lapack.data(), H_lapack.data(), mat * sizeof(cdouble));

        // 1) Eigenvalue comparison.
        double max_eig_rel = 0.0;
        for (int i = 0; i < kN; ++i) {
            const double scale = std::max(std::abs(W_lapack[i]), 1.0);
            const double rel   = std::abs(W_gpu[i] - W_lapack[i]) / scale;
            max_eig_rel = std::max(max_eig_rel, rel);
        }

        // 2) Subspace metric per column.
        // For each i, inner = |sum_k conj(V_gpu[k,i]) * V_lapack[k,i]|.
        // phase_error = 1 - inner. Track which indices sit in a near-degenerate
        // eigenvalue cluster (gap < kDegenGap), and exclude them from the worst-
        // case but still flag.
        double max_phase_nondegen = 0.0;
        int    degen_excluded     = 0;
        double max_phase_in_degen = 0.0;
        for (int i = 0; i < kN; ++i) {
            cdouble inner(0.0, 0.0);
            for (int k = 0; k < kN; ++k) {
                const cdouble vg = V_gpu   [static_cast<std::size_t>(i) * kN + k];
                const cdouble vl = V_lapack[static_cast<std::size_t>(i) * kN + k];
                inner += std::conj(vg) * vl;
            }
            const double phase_err = 1.0 - std::abs(inner);

            // Determine degeneracy: gap to nearest neighbour eigenvalue.
            double gap_left  = (i > 0)        ? std::abs(W_lapack[i] - W_lapack[i - 1]) : INFINITY;
            double gap_right = (i < kN - 1)   ? std::abs(W_lapack[i + 1] - W_lapack[i]) : INFINITY;
            const double gap = std::min(gap_left, gap_right);
            if (gap < kDegenGap) {
                ++degen_excluded;
                max_phase_in_degen = std::max(max_phase_in_degen, phase_err);
            } else {
                max_phase_nondegen = std::max(max_phase_nondegen, phase_err);
            }
        }

        max_eig_rel_overall = std::max(max_eig_rel_overall, max_eig_rel);
        max_phase_overall   = std::max(max_phase_overall,   max_phase_nondegen);

        const bool ok = (max_eig_rel < kTolEig) && (max_phase_nondegen < kTolVec);
        std::printf("matrix %d: eig_rel=%.3e, phase_err=%.3e",
                    b, max_eig_rel, max_phase_nondegen);
        if (degen_excluded > 0) {
            std::printf(" (degen_excluded=%d, max_phase_in_degen=%.3e)",
                        degen_excluded, max_phase_in_degen);
        }
        std::printf("  %s\n", ok ? "PASS" : "FAIL");
        if (ok) ++pass_count;
    }

    std::printf("SUMMARY: %d/%d passed, max_eig_rel_overall=%.3e, max_phase_overall=%.3e\n",
                pass_count, fx.B, max_eig_rel_overall, max_phase_overall);

    CUDA_CHECK(cudaFree(d_H)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_info));
    return (pass_count == fx.B) ? 0 : 1;
}
