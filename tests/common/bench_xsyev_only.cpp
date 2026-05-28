// standalone cuSOLVER XsyevBatched benchmark.
//
// Times cusolverDnXsyevBatched alone (no Cholesky, no trsm, no Sunny) on a
// 1024-matrix N=54 ComplexF64 Hermitian batch. The matrices are made
// Hermitian via H = A + A^H so they're suitable for the eigenvalue solver
// directly. This isolates cuSOLVER's batched eigensolver throughput at
// our matrix size and lets us compare against our cuSolverDx Heev-only
// timing (reduce-only / no-vectors variants gave
// heev-no-vec at 0.0667 s for 1024 mats = 65 µs/mat).

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
#include <cstring>
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
    std::printf("cuSOLVER XsyevBatched standalone — "
                "N=%d, batch=%d, samples=%d\n", kN, kBatch, kSamples);

    // Generate Hermitian inputs on host.
    const std::size_t mat = (std::size_t)kN * kN;
    std::vector<cdouble> H_host(mat * kBatch);
    {
        std::mt19937_64 rng(kSeed);
        for (int b = 0; b < kBatch; ++b) {
            make_hermitian(kN, rng, &H_host[mat * b]);
        }
    }

    // Set up cuSOLVER.
    cusolverDnHandle_t handle = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cusolverDnParams_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    cuDoubleComplex* d_A = nullptr;
    double*          d_W = nullptr;
    int*             d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,    mat * kBatch * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_W,    (std::size_t)kN * kBatch * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, kBatch * sizeof(int)));

    // Workspace size query.
    std::size_t ws_gpu_bytes = 0, ws_cpu_bytes = 0;
    CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
        handle,
        params,
        CUSOLVER_EIG_MODE_VECTOR,    // 'V' — overwrite_vectors equivalent
        CUBLAS_FILL_MODE_LOWER,      // 'L'
        (int64_t)kN,
        CUDA_C_64F,                  // dataTypeA = ComplexF64
        d_A, (int64_t)kN,
        CUDA_R_64F,                  // dataTypeW = real Float64
        d_W,
        CUDA_C_64F,                  // computeType
        &ws_gpu_bytes,
        &ws_cpu_bytes,
        (int64_t)kBatch));
    std::printf("XsyevBatched workspace: gpu=%zu B, cpu=%zu B\n",
                ws_gpu_bytes, ws_cpu_bytes);

    void* d_work = nullptr;
    if (ws_gpu_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_work, ws_gpu_bytes));
    }
    std::vector<unsigned char> h_work(std::max(ws_cpu_bytes, (std::size_t)1));

    auto reload_inputs = [&]() {
        CUDA_CHECK(cudaMemcpy(d_A, H_host.data(),
                              mat * kBatch * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_info, 0, kBatch * sizeof(int)));
    };

    auto run_one = [&]() -> double {
        // Reload between samples — XsyevBatched is in-place (overwrites
        // A with eigenvectors), so we need fresh inputs each sample to
        // make the work identical.
        reload_inputs();
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0, 0);
        CUSOLVER_CHECK(cusolverDnXsyevBatched(
            handle,
            params,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            (int64_t)kN,
            CUDA_C_64F,
            d_A, (int64_t)kN,
            CUDA_R_64F,
            d_W,
            CUDA_C_64F,
            d_work, ws_gpu_bytes,
            h_work.data(), ws_cpu_bytes,
            d_info, (int64_t)kBatch));
        cudaEventRecord(e1, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        return (double)ms / 1000.0;
    };

    // LAPACK reference for batch idx 0.
    auto ref_eigvals = lapack_reference_eigvals(H_host.data(), kN);

    // Warm-up.
    run_one();

    // Correctness check after warm-up reload+run.
    run_one();
    std::vector<double> got_eigvals_b0(kN);
    CUDA_CHECK(cudaMemcpy(got_eigvals_b0.data(), d_W,
                          kN * sizeof(double), cudaMemcpyDeviceToHost));
    double rel_err = max_rel_eig_diff(ref_eigvals, got_eigvals_b0);
    std::printf("Correctness (batch idx 0 vs LAPACK ZHEEV):\n");
    std::printf("  max relative eigenvalue diff = %.3e\n", rel_err);

    std::vector<double> samples;
    for (int s = 0; s < kSamples; ++s) {
        samples.push_back(run_one());
    }
    double med = median_of(samples);
    double lo  = *std::min_element(samples.begin(), samples.end());
    double hi  = *std::max_element(samples.begin(), samples.end());
    double mps = (double)kBatch / med;

    std::printf("\n--- cusolverDnXsyevBatched (V, L, ComplexF64, N=%d, batch=%d) ---\n", kN, kBatch);
    std::printf("  samples (s): ");
    for (double t : samples) std::printf("%.5f ", t);
    std::printf("\n  median %.5f s, range %.5f-%.5f s, %.0f mat/sec\n",
                med, lo, hi, mps);
    std::printf("  per-matrix: %.2f µs\n", med * 1e6 / kBatch);

    // Compare against our heev-with-vectors timing from
    // test_heev_fraction.cpp (full - reduce = 0.11273 s, but that
    // includes back-trsm; cleaner number is full - no-vectors heev
    // call boundary). The most apples-to-apples comparison is:
    //   T_xsyev_with_vec  vs  T_full - T_reduce - T_back_trsm
    // Since we can't isolate T_back_trsm without another variant,
    // we simply print our heev-no-vec time (0.06670 s) and the
    // full kernel time (0.12661 s) for context.
    constexpr double kOurHeevNoVec_s = 0.06670;        // from diagnostic
    constexpr double kOurFull_s      = 0.12661;
    constexpr double kOurReduce_s    = 0.01388;
    std::printf("\nContext (measured numbers, same hardware/workload):\n");
    std::printf("  our reduce-only   (potrf+trsm×2)              : %.5f s   (~%.0f mat/s)\n",
                kOurReduce_s, kBatch / kOurReduce_s);
    std::printf("  our heev no-vec   (potrf+trsm×2 + heev no-vec): %.5f s\n", kOurHeevNoVec_s);
    std::printf("  our full          (5-stage pipeline)          : %.5f s   (~%.0f mat/s)\n",
                kOurFull_s, kBatch / kOurFull_s);
    std::printf("\n  our heev-with-vec (full - reduce)             : %.5f s   (~%.0f mat/s if isolated)\n",
                kOurFull_s - kOurReduce_s, kBatch / (kOurFull_s - kOurReduce_s));
    std::printf("  XsyevBatched (this run, with vectors)         : %.5f s   (~%.0f mat/s)\n",
                med, mps);
    const double our_heev_with_vec = kOurFull_s - kOurReduce_s;
    std::printf("\nRatio: cuSOLVER XsyevBatched / our heev-with-vec = %.3fx\n",
                our_heev_with_vec / med);
    std::printf("  (>1 means XsyevBatched is faster than our heev for the eigensolver work alone.)\n");

    if (d_work) cudaFree(d_work);
    cudaFree(d_A); cudaFree(d_W); cudaFree(d_info);
    cusolverDnDestroyParams(params);
    cusolverDnDestroy(handle);
    return 0;
}
