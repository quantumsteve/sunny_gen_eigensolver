// Step 2b test: validate geneig_reduce_n54_kernel against
// LAPACKE_zpotrf + LAPACKE_zhegst (which is the canonical M = L^{-1} H L^{-H}
// reduction in LAPACK).
//
// Loads tests/data/n54_b8_s42_random.bin (8 generic 54x54 H/S pairs), runs the
// GPU reduction kernel, computes the LAPACK reference, and reports max abs /
// max rel difference between the two M matrices (lower triangle only — strict
// upper is zeroed in both before comparison). Pass criterion: rel_diff < 1e-11.

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

extern "C" void geneig_reduce_n54_launch(const cuDoubleComplex* d_H,
                                         const cuDoubleComplex* d_S,
                                         cuDoubleComplex*       d_M,
                                         int*                   d_info,
                                         cudaStream_t           stream);
extern "C" unsigned int geneig_reduce_n54_block_dim_x();
extern "C" unsigned int geneig_reduce_n54_shared_mem_bytes();

namespace {

constexpr std::int32_t kMagic   = 0x47454947;
constexpr std::int32_t kVersion = 1;
constexpr int          kN       = 54;
constexpr double       kTol     = 1e-11;

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

void zero_strict_upper(cdouble* M, int n) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < j; ++i)
            M[static_cast<std::size_t>(j) * n + i] = cdouble(0.0, 0.0);
}

}  // namespace

int main() {
    const Fixture fx = load_fixture("tests/data/n54_b8_s42_random.bin");
    std::printf("loaded fixture: B=%d, N=%d\n", fx.B, kN);
    std::printf("kernel block_dim.x=%u, shared_mem_bytes=%u\n",
                geneig_reduce_n54_block_dim_x(),
                geneig_reduce_n54_shared_mem_bytes());

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;

    cuDoubleComplex* d_H = nullptr; cuDoubleComplex* d_S = nullptr;
    cuDoubleComplex* d_M = nullptr; int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_H,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_S,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_M,    mat * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    std::vector<cdouble> M_gpu(mat), H_lapack(mat), S_lapack(mat);
    int    pass_count = 0;
    double max_rel_overall = 0.0;

    for (int b = 0; b < fx.B; ++b) {
        const cdouble* H_b = &fx.H[mat * b];
        const cdouble* S_b = &fx.S[mat * b];

        // GPU path.
        CUDA_CHECK(cudaMemcpy(d_H, H_b, mat * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_S, S_b, mat * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));
        geneig_reduce_n54_launch(d_H, d_S, d_M, d_info, /*stream=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());

        int gpu_info = -1;
        CUDA_CHECK(cudaMemcpy(&gpu_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(M_gpu.data(), d_M, mat * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        if (gpu_info != 0) {
            std::printf("matrix %d: GPU info=%d  FAIL (potrf reported failure)\n", b, gpu_info);
            continue;
        }

        // LAPACK reference: zpotrf(S) → L, then zhegst(itype=1, 'L', N, H, lda, L, ldL).
        // zhegst overwrites H with M = L^{-1} H L^{-H}, only touching the
        // lower triangle (per uplo='L'). The strict upper triangle of H is
        // left as the original input data, which we then zero for comparison.
        std::memcpy(S_lapack.data(), S_b, mat * sizeof(cdouble));
        std::memcpy(H_lapack.data(), H_b, mat * sizeof(cdouble));

        lapack_int li = LAPACKE_zpotrf(
            LAPACK_COL_MAJOR, 'L', kN,
            reinterpret_cast<lapack_complex_double*>(S_lapack.data()), kN);
        if (li != 0) {
            std::printf("matrix %d: LAPACK zpotrf info=%d  FAIL\n", b, (int)li);
            continue;
        }
        li = LAPACKE_zhegst(
            LAPACK_COL_MAJOR, /*itype=*/1, 'L', kN,
            reinterpret_cast<lapack_complex_double*>(H_lapack.data()), kN,
            reinterpret_cast<lapack_complex_double*>(S_lapack.data()), kN);
        if (li != 0) {
            std::printf("matrix %d: LAPACK zhegst info=%d  FAIL\n", b, (int)li);
            continue;
        }
        zero_strict_upper(H_lapack.data(), kN);

        double max_abs = 0.0, frob = 0.0;
        for (std::size_t k = 0; k < mat; ++k) {
            double diff = std::abs(M_gpu[k] - H_lapack[k]);
            max_abs = std::max(max_abs, diff);
            frob += std::norm(H_lapack[k]);
        }
        frob = std::sqrt(frob);
        double max_rel = (frob > 0.0) ? (max_abs / frob) : max_abs;
        max_rel_overall = std::max(max_rel_overall, max_rel);
        const bool ok = (max_rel < kTol);
        std::printf("matrix %d: max_abs_diff=%.3e, max_rel_diff=%.3e  %s\n",
                    b, max_abs, max_rel, ok ? "PASS" : "FAIL");
        if (ok) ++pass_count;
    }

    std::printf("SUMMARY: %d/%d passed, max_rel_diff_overall=%.3e\n",
                pass_count, fx.B, max_rel_overall);
    CUDA_CHECK(cudaFree(d_H)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_M)); CUDA_CHECK(cudaFree(d_info));
    return (pass_count == fx.B) ? 0 : 1;
}
