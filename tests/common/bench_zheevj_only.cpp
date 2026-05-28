// Steven-comparison — standalone cuSOLVER ZheevjBatched (Jacobi) benchmark.
//
// Mirrors bench_xsyev_only.cpp on identical inputs (random complex Hermitian
// at N=54, batch=1024, ComplexF64) but uses the legacy Jacobi-based
// cusolverDnZheevjBatched. Steven Hahn asked how this compares to the modern
// XsyevBatched and to our cuSolverDx::Heev path; this gives us numbers to
// answer that.
//
// Defaults for the Jacobi solver (cuSOLVER 12.x): tolerance ~1e-7,
// max_sweeps ~15. We deliberately do NOT tighten these — the question is
// "what does the user get out of the box at N=54?".

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <lapacke.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr int kN       = 54;
constexpr int kBatch   = 1024;
constexpr int kSamples = 5;
constexpr std::uint64_t kSeed = 0xC0FFEE0042ULL;

#define CUDA_CHECK(expr) do {                                          \
    cudaError_t _r = (expr);                                           \
    if (_r != cudaSuccess) {                                           \
        std::fprintf(stderr, "CUDA error at %s: %s\n", #expr,          \
                     cudaGetErrorString(_r));                          \
        std::exit(20);                                                 \
    }                                                                  \
} while (0)

#define CUSOLVER_CHECK(expr) do {                                      \
    cusolverStatus_t _r = (expr);                                      \
    if (_r != CUSOLVER_STATUS_SUCCESS) {                               \
        std::fprintf(stderr, "cuSOLVER error at %s: %d\n", #expr,      \
                     (int)_r);                                         \
        std::exit(21);                                                 \
    }                                                                  \
} while (0)

void make_hermitian(int n, std::mt19937_64& rng, cdouble* H) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble> A((std::size_t)n * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[(std::size_t)j * n + i] = cdouble(g(rng), g(rng));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            cdouble aij = A[(std::size_t)j * n + i];
            cdouble aji = A[(std::size_t)i * n + j];
            H[(std::size_t)j * n + i] = aij + std::conj(aji);
        }
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// LAPACK zheev reference for batch index 0. Returns sorted eigenvalues.
std::vector<double> lapack_reference_eigvals(const cdouble* H, int n) {
    std::vector<cdouble> work(static_cast<std::size_t>(n) * n);
    std::memcpy(work.data(), H, sizeof(cdouble) * n * n);
    std::vector<double> w(n);
    lapack_int info = LAPACKE_zheev(
        LAPACK_COL_MAJOR, 'N', 'L', n,
        reinterpret_cast<lapack_complex_double*>(work.data()), n,
        w.data());
    if (info != 0) {
        std::fprintf(stderr, "LAPACKE_zheev reference failed: info=%d\n", (int)info);
        std::exit(30);
    }
    return w;
}

double max_rel_eig_diff(const std::vector<double>& ref,
                        const std::vector<double>& got) {
    auto a = ref; auto b = got;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::abs(a[i] - b[i]) / std::max(std::abs(a[i]), 1.0);
        if (d > m) m = d;
    }
    return m;
}

}  // namespace

int main() {
    std::printf("Steven-comparison cuSOLVER ZheevjBatched standalone — "
                "N=%d, batch=%d, samples=%d\n", kN, kBatch, kSamples);

    const std::size_t mat = (std::size_t)kN * kN;
    std::vector<cdouble> H_host(mat * kBatch);
    {
        std::mt19937_64 rng(kSeed);
        for (int b = 0; b < kBatch; ++b) {
            make_hermitian(kN, rng, &H_host[mat * b]);
        }
    }

    // LAPACK reference eigenvalues for batch index 0 (correctness check).
    auto ref_eigvals = lapack_reference_eigvals(H_host.data(), kN);

    cusolverDnHandle_t handle = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // Jacobi parameters — explicit defaults so the report records what we used.
    syevjInfo_t syevj_params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
    constexpr double kTolerance = 1.0e-7;
    constexpr int    kMaxSweeps = 15;
    constexpr int    kSortEig   = 1;
    CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, kTolerance));
    CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, kMaxSweeps));
    CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, kSortEig));
    std::printf("ZheevjBatched params: tol=%.0e, max_sweeps=%d, sort_eig=%d\n",
                kTolerance, kMaxSweeps, kSortEig);

    cuDoubleComplex* d_A    = nullptr;
    double*          d_W    = nullptr;
    int*             d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,    mat * kBatch * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_W,    (std::size_t)kN * kBatch * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, kBatch * sizeof(int)));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(
        handle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        kN,
        d_A, kN,
        d_W,
        &lwork,
        syevj_params,
        kBatch));
    const std::size_t ws_bytes = (std::size_t)lwork * sizeof(cuDoubleComplex);
    std::printf("ZheevjBatched workspace: %d cuDoubleComplex elements (%zu B = %.2f MB)\n",
                lwork, ws_bytes, ws_bytes / (1024.0 * 1024.0));

    cuDoubleComplex* d_work = nullptr;
    if (lwork > 0) {
        CUDA_CHECK(cudaMalloc(&d_work, ws_bytes));
    }

    auto reload_inputs = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, H_host.data(),
                              mat * kBatch * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_info, 0, kBatch * sizeof(int)));
    };

    auto run_one = [&]() -> double {
        reload_inputs();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, 0);
        CUSOLVER_CHECK(cusolverDnZheevjBatched(
            handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            kN,
            d_A, kN,
            d_W,
            d_work, lwork,
            d_info,
            syevj_params,
            kBatch));
        cudaEventRecord(e1, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        return (double)ms / 1000.0;
    };

    // Warm-up.
    run_one();

    // Correctness: pull eigenvalues from batch 0 after the warm-up's reload+run,
    // then re-run to ensure timing samples reflect the same call sequence.
    run_one();
    std::vector<double> got_eigvals_b0(kN);
    CUDA_CHECK(cudaMemcpy(got_eigvals_b0.data(), d_W,
                          kN * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<int> info_host(kBatch);
    CUDA_CHECK(cudaMemcpy(info_host.data(), d_info,
                          kBatch * sizeof(int), cudaMemcpyDeviceToHost));
    int n_failed = 0;
    for (int i : info_host) if (i != 0) ++n_failed;
    double rel_err = max_rel_eig_diff(ref_eigvals, got_eigvals_b0);
    std::printf("Correctness (batch idx 0 vs LAPACK ZHEEV):\n");
    std::printf("  max relative eigenvalue diff = %.3e\n", rel_err);
    std::printf("  Jacobi info != 0 across batch: %d / %d\n",
                n_failed, kBatch);
    if (n_failed > 0) {
        std::printf("  (info > 0 means did not converge in %d sweeps; ", kMaxSweeps);
        std::printf("info = sweep counts attained for failed batches.)\n");
        // Print first few failure codes.
        int shown = 0;
        for (int i = 0; i < kBatch && shown < 5; ++i) {
            if (info_host[i] != 0) {
                std::printf("    batch=%d info=%d\n", i, info_host[i]);
                ++shown;
            }
        }
    }

    std::vector<double> samples;
    for (int s = 0; s < kSamples; ++s) {
        samples.push_back(run_one());
    }
    double med = median_of(samples);
    double lo  = *std::min_element(samples.begin(), samples.end());
    double hi  = *std::max_element(samples.begin(), samples.end());
    double mps = (double)kBatch / med;

    std::printf("\n--- cusolverDnZheevjBatched (V, L, ComplexF64, N=%d, batch=%d) ---\n", kN, kBatch);
    std::printf("  samples (s): ");
    for (double t : samples) std::printf("%.5f ", t);
    std::printf("\n  median %.5f s, range %.5f-%.5f s, %.0f mat/sec\n",
                med, lo, hi, mps);
    std::printf("  per-matrix: %.2f us\n", med * 1e6 / kBatch);
    std::printf("  workspace : %zu B = %.3f MB total (across batch)\n",
                ws_bytes, ws_bytes / (1024.0 * 1024.0));
    std::printf("              per matrix: %zu B = %.2f KB\n",
                ws_bytes / kBatch, (ws_bytes / kBatch) / 1024.0);

    if (d_work) cudaFree(d_work);
    cudaFree(d_A); cudaFree(d_W); cudaFree(d_info);
    cusolverDnDestroySyevjInfo(syevj_params);
    cusolverDnDestroy(handle);
    return 0;
}
