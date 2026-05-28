// Step 2e test: validate geneig_full_n54_batched_kernel against
// (a) the n54_b32_s123_random fixture for batch_size in {1, 8, 32}, and
// (b) a synthetic 1024-matrix batch generated/validated against LAPACK zhegv.
//
// Comparison metrics (per matrix) are identical to test_full_n54:
//   eig_rel  = max_i |W_gpu[i] - W_ref[i]| / max(|W_ref[i]|, 1)
//   phase_err = max_i (1 - |u_gpu[:,i]^H * S * u_ref[:,i]|)   (S-inner-product)
// Both tolerances 1e-10. Near-degenerate eigenvalue clusters (gap < kDegenGap)
// are flagged but not gated.

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <lapacke.h>

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
#include <vector>

using cdouble = std::complex<double>;

extern "C" void geneig_full_n54_batched_launch(const cuDoubleComplex* d_H,
                                               const cuDoubleComplex* d_S,
                                               double*                d_W,
                                               cuDoubleComplex*       d_U,
                                               int*                   d_info,
                                               int                    batch_size,
                                               cudaStream_t           stream);
extern "C" unsigned int geneig_full_n54_batched_block_dim_x();
extern "C" unsigned int geneig_full_n54_batched_shared_mem_bytes();

namespace {

constexpr std::int32_t kMagic    = 0x47454947;
constexpr std::int32_t kVersion  = 1;
constexpr int          kN        = 54;
constexpr double       kTolEig   = 1e-10;
constexpr double       kTolVec   = 1e-10;
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

void mul_S_v(const cdouble* S, const cdouble* v, cdouble* Sv, int n) {
    for (int row = 0; row < n; ++row) {
        cdouble acc(0.0, 0.0);
        for (int col = 0; col < n; ++col) {
            acc += S[static_cast<std::size_t>(col) * n + row] * v[col];
        }
        Sv[row] = acc;
    }
}

struct PerMatResult { double eig_rel = 0.0; double phase_err = 0.0; int degen_excluded = 0; };

PerMatResult compare_one(const cdouble* S_b,
                         const cdouble* U_ref, const double* W_ref,
                         const cdouble* U_gpu, const double* W_gpu) {
    PerMatResult r;
    for (int i = 0; i < kN; ++i) {
        const double scale = std::max(std::abs(W_ref[i]), 1.0);
        const double rel   = std::abs(W_gpu[i] - W_ref[i]) / scale;
        r.eig_rel = std::max(r.eig_rel, rel);
    }
    std::vector<cdouble> Sv(kN);
    for (int i = 0; i < kN; ++i) {
        const cdouble* u_gpu = &U_gpu[static_cast<std::size_t>(i) * kN];
        const cdouble* u_ref = &U_ref[static_cast<std::size_t>(i) * kN];
        mul_S_v(S_b, u_ref, Sv.data(), kN);
        cdouble inner(0.0, 0.0);
        for (int k = 0; k < kN; ++k) inner += std::conj(u_gpu[k]) * Sv[k];
        const double phase_err = 1.0 - std::abs(inner);

        const double gap_left  = (i > 0)      ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
        const double gap_right = (i < kN - 1) ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
        const double gap = std::min(gap_left, gap_right);
        if (gap < kDegenGap) ++r.degen_excluded;
        else                 r.phase_err = std::max(r.phase_err, phase_err);
    }
    return r;
}

// Run a batch_size-matrix kernel launch on a contiguous slice [start..start+batch_size)
// of the supplied fixture. Returns true if all matrices in the slice pass.
bool run_batch_against_fixture(const Fixture& fx, int start, int batch_size,
                               cuDoubleComplex* d_H, cuDoubleComplex* d_S,
                               cuDoubleComplex* d_U, double* d_W, int* d_info,
                               const char* tag) {
    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    CUDA_CHECK(cudaMemcpy(d_H, &fx.H[mat * start],
                          mat * batch_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, &fx.S[mat * start],
                          mat * batch_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_info, 0, batch_size * sizeof(int)));

    geneig_full_n54_batched_launch(d_H, d_S, d_W, d_U, d_info, batch_size, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> info(batch_size, 0);
    std::vector<cdouble> U_gpu(mat * batch_size);
    std::vector<double>  W_gpu(static_cast<std::size_t>(kN) * batch_size);
    CUDA_CHECK(cudaMemcpy(info.data(), d_info, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(W_gpu.data(), d_W, batch_size * kN * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(U_gpu.data(), d_U, mat * batch_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    int    pass_count          = 0;
    double max_eig_rel_overall = 0.0;
    double max_phase_overall   = 0.0;
    int    total_degen         = 0;

    for (int b = 0; b < batch_size; ++b) {
        const int gi = start + b;  // global fixture index
        if (info[b] != 0) {
            std::printf("    [%s] matrix %d (fixture idx %d): info=%d  FAIL\n",
                        tag, b, gi, info[b]);
            continue;
        }
        const cdouble* S_b   = &fx.S[mat * gi];
        const cdouble* U_ref = &fx.U[mat * gi];
        const double*  W_ref = &fx.W[static_cast<std::size_t>(kN) * gi];
        const cdouble* U_g   = &U_gpu[mat * b];
        const double*  W_g   = &W_gpu[static_cast<std::size_t>(kN) * b];

        const PerMatResult r = compare_one(S_b, U_ref, W_ref, U_g, W_g);
        max_eig_rel_overall = std::max(max_eig_rel_overall, r.eig_rel);
        max_phase_overall   = std::max(max_phase_overall,   r.phase_err);
        total_degen         += r.degen_excluded;

        const bool ok = (r.eig_rel < kTolEig) && (r.phase_err < kTolVec);
        if (!ok) {
            std::printf("    [%s] matrix %d (fixture idx %d): eig_rel=%.3e phase_err=%.3e  FAIL\n",
                        tag, b, gi, r.eig_rel, r.phase_err);
        }
        if (ok) ++pass_count;
    }
    std::printf("    [%s] batch_size=%d start=%d: %d/%d passed, "
                "max_eig_rel=%.3e, max_phase=%.3e, degen_excluded=%d\n",
                tag, batch_size, start, pass_count, batch_size,
                max_eig_rel_overall, max_phase_overall, total_degen);
    return pass_count == batch_size;
}

// === Synthetic 1024-matrix test =======================================

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

bool test_large_batch(unsigned int batch_size, std::uint64_t seed) {
    std::printf("\n--- synthetic large-batch test: B=%u, seed=%lu ---\n",
                batch_size, (unsigned long)seed);
    const std::size_t mat = static_cast<std::size_t>(kN) * kN;

    std::vector<cdouble> H_all(mat * batch_size);
    std::vector<cdouble> S_all(mat * batch_size);
    std::vector<cdouble> U_ref(mat * batch_size);
    std::vector<double>  W_ref(static_cast<std::size_t>(kN) * batch_size);

    std::mt19937_64 rng(seed);
    for (unsigned b = 0; b < batch_size; ++b) {
        make_hermitian(kN, rng, &H_all[mat * b]);
        make_hpd_random(kN, rng, &S_all[mat * b]);
    }

    auto t_lapack_start = std::chrono::steady_clock::now();
    for (unsigned b = 0; b < batch_size; ++b) {
        std::vector<cdouble> H_w(mat), S_w(mat);
        std::memcpy(H_w.data(), &H_all[mat * b], sizeof(cdouble) * mat);
        std::memcpy(S_w.data(), &S_all[mat * b], sizeof(cdouble) * mat);
        lapack_int li = LAPACKE_zhegv(
            LAPACK_COL_MAJOR, /*itype=*/1, 'V', 'L', kN,
            reinterpret_cast<lapack_complex_double*>(H_w.data()), kN,
            reinterpret_cast<lapack_complex_double*>(S_w.data()), kN,
            &W_ref[static_cast<std::size_t>(kN) * b]);
        if (li != 0) {
            std::fprintf(stderr, "zhegv failed at b=%u, info=%d\n", b, (int)li);
            return false;
        }
        std::memcpy(&U_ref[mat * b], H_w.data(), sizeof(cdouble) * mat);
    }
    auto t_lapack_end = std::chrono::steady_clock::now();
    const double t_lapack_s =
        std::chrono::duration<double>(t_lapack_end - t_lapack_start).count();

    cuDoubleComplex* d_H = nullptr; cuDoubleComplex* d_S = nullptr;
    cuDoubleComplex* d_U = nullptr; double* d_W = nullptr; int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_H,    mat * batch_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_S,    mat * batch_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_U,    mat * batch_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_W,    static_cast<std::size_t>(kN) * batch_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, batch_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_H, H_all.data(), mat * batch_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S, S_all.data(), mat * batch_size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_info, 0, batch_size * sizeof(int)));

    // Warmup launch (separate batch_size=1 to absorb static attr-set + JIT cache).
    geneig_full_n54_batched_launch(d_H, d_S, d_W, d_U, d_info, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    cudaEventRecord(e0, 0);
    geneig_full_n54_batched_launch(d_H, d_S, d_W, d_U, d_info,
                                   static_cast<int>(batch_size), /*stream=*/0);
    cudaEventRecord(e1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    const double t_gpu_s = ms / 1000.0;

    std::vector<int> info(batch_size, 0);
    std::vector<cdouble> U_gpu(mat * batch_size);
    std::vector<double>  W_gpu(static_cast<std::size_t>(kN) * batch_size);
    CUDA_CHECK(cudaMemcpy(info.data(), d_info, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(W_gpu.data(), d_W, batch_size * kN * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(U_gpu.data(), d_U, mat * batch_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    int pass_count = 0;
    double max_eig_rel_overall = 0.0;
    double max_phase_overall   = 0.0;
    int    total_degen         = 0;

    for (unsigned b = 0; b < batch_size; ++b) {
        if (info[b] != 0) {
            std::printf("    matrix %u: info=%d  FAIL\n", b, info[b]);
            continue;
        }
        const cdouble* S_b   = &S_all[mat * b];
        const cdouble* U_r   = &U_ref[mat * b];
        const double*  W_r   = &W_ref[static_cast<std::size_t>(kN) * b];
        const cdouble* U_g   = &U_gpu[mat * b];
        const double*  W_g   = &W_gpu[static_cast<std::size_t>(kN) * b];

        const PerMatResult r = compare_one(S_b, U_r, W_r, U_g, W_g);
        max_eig_rel_overall = std::max(max_eig_rel_overall, r.eig_rel);
        max_phase_overall   = std::max(max_phase_overall,   r.phase_err);
        total_degen         += r.degen_excluded;
        const bool ok = (r.eig_rel < kTolEig) && (r.phase_err < kTolVec);
        if (!ok) {
            std::printf("    matrix %u: eig_rel=%.3e phase_err=%.3e  FAIL\n",
                        b, r.eig_rel, r.phase_err);
        }
        if (ok) ++pass_count;
    }

    CUDA_CHECK(cudaFree(d_H)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_U)); CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_info));

    const double throughput_mat_per_s = batch_size / t_gpu_s;
    std::printf("    SUMMARY: %d/%u passed, max_eig_rel=%.3e, max_phase=%.3e, degen_excluded=%d\n",
                pass_count, batch_size, max_eig_rel_overall, max_phase_overall, total_degen);
    std::printf("    timing: GPU batched kernel = %.3f s   (%.0f matrices/sec)\n",
                t_gpu_s, throughput_mat_per_s);
    std::printf("    timing: LAPACK zhegv host  = %.3f s   (%.0f matrices/sec)\n",
                t_lapack_s, batch_size / t_lapack_s);
    return static_cast<unsigned>(pass_count) == batch_size;
}

}  // namespace

int main() {
    std::printf("kernel block_dim.x=%u, total_smem_bytes=%u\n",
                geneig_full_n54_batched_block_dim_x(),
                geneig_full_n54_batched_shared_mem_bytes());

    // ----- Fixture-based tests --------------------------------------------
    const Fixture fx = load_fixture("tests/data/n54_b32_s123_random.bin");
    std::printf("\n--- fixture: tests/data/n54_b32_s123_random.bin (B=%d) ---\n", fx.B);

    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    cuDoubleComplex* d_H = nullptr; cuDoubleComplex* d_S = nullptr;
    cuDoubleComplex* d_U = nullptr; double* d_W = nullptr; int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_H,    mat * fx.B * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_S,    mat * fx.B * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_U,    mat * fx.B * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_W,    static_cast<std::size_t>(kN) * fx.B * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, fx.B * sizeof(int)));

    bool ok_b1  = run_batch_against_fixture(fx, /*start=*/0, /*batch=*/1,
                                            d_H, d_S, d_U, d_W, d_info, "B=1 ");
    bool ok_b8  = run_batch_against_fixture(fx, /*start=*/0, /*batch=*/8,
                                            d_H, d_S, d_U, d_W, d_info, "B=8 ");
    bool ok_b32 = run_batch_against_fixture(fx, /*start=*/0, /*batch=*/32,
                                            d_H, d_S, d_U, d_W, d_info, "B=32");

    CUDA_CHECK(cudaFree(d_H)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_U)); CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_info));

    // ----- Synthetic 1024-matrix test -------------------------------------
    bool ok_b1024 = test_large_batch(/*batch_size=*/1024, /*seed=*/0xC0FFEE0042ULL);

    const bool overall = ok_b1 && ok_b8 && ok_b32 && ok_b1024;
    std::printf("\nOVERALL: %s\n", overall ? "ALL PASSED" : "SOME FAILED");
    return overall ? 0 : 1;
}
