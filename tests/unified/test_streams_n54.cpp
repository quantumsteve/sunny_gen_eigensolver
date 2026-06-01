// multi-stream chunked launch benchmark.
//
// Drives UnifiedGeneigSolver::launch_chunked across stream configurations
// {1, 2, 4, 8} and reports median throughput. The benchmark allocates a
// single device-side workspace sized for `total_matrices` and dispatches
// 30 chunks of `chunk_size` matrices each round-robined over the chosen
// number of streams. HtoD and DtoH happen outside the timed region; what
// we measure is the kernel-stream concurrency only.
//
// Memory note: a 360,000-matrix workload at N=54 would require >50 GB of
// device memory (H+S+U at fp64 complex). On the development laptop GPU
// (12 GB) we scale the chunk_size down so 30 chunks fit. Production Sunny
// uses chunk_size=12000; we run with chunk_size=2000 to keep the 30-chunk
// structure on a memory-constrained GPU. The streaming pattern is
// identical; only the per-chunk batch size differs.
//
// Correctness check: 8 spot-sampled matrices are validated against LAPACK
// for each (stream config, sample) pair via the S-inner-product / phase
// metric used in test_nvrtc_n54.cpp.

#include <cuComplex.h>
#include <cuda.h>

#include "unified/unified_solver.hpp"

#include "mkl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr int    kN          = 54;
constexpr int    kChunkSize  = 2000;
constexpr int    kNumChunks  = 30;
constexpr int    kTotalMats  = kChunkSize * kNumChunks;     // 60,000
constexpr int    kSamples    = 5;
constexpr int    kSpotCount  = 8;
constexpr double kTolEig     = 1e-10;
constexpr double kTolVec     = 1e-10;
constexpr double kDegenGap   = 1e-6;

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

// Generate a 12000-matrix base set, then replicate to fill kTotalMats.
// Replication is fine for throughput measurement; correctness validation
// uses the original 12000 reference results.
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

// LAPACK reference for a single (H, S) pair; W ascending, U columns are
// generalized eigenvectors S-orthonormalized. Same convention as // fixtures (zhegv with itype=1, jobz='V', uplo='L', column-major).
struct LapackResult {
    std::vector<double>  W;
    std::vector<cdouble> U;
};

LapackResult lapack_zhegv(const cdouble* H, const cdouble* S, int n) {
    LapackResult r;
    r.W.resize(n);
    r.U.assign(H, H + static_cast<std::size_t>(n) * n);
    std::vector<cdouble> S_copy(S, S + static_cast<std::size_t>(n) * n);
    lapack_int info = LAPACKE_zhegv(
        LAPACK_COL_MAJOR, /*itype=*/1, /*jobz=*/'V', /*uplo=*/'L', n,
        reinterpret_cast<lapack_complex_double*>(r.U.data()), n,
        reinterpret_cast<lapack_complex_double*>(S_copy.data()), n,
        r.W.data());
    if (info != 0) {
        std::fprintf(stderr, "LAPACKE_zhegv info=%d\n", static_cast<int>(info));
        std::exit(31);
    }
    return r;
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

struct Sample { double mps = 0.0; double max_eig_rel = 0.0; double max_phase = 0.0; };

struct ConfigResult {
    int                 num_streams = 0;
    std::vector<double> mps_samples;
    double              median_mps  = 0.0;
    double              min_mps     = 0.0;
    double              max_mps     = 0.0;
    double              max_eig_rel = 0.0;
    double              max_phase   = 0.0;
    bool                pass        = false;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if ((n & 1u) == 0u) return 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return v[n / 2];
}

}  // namespace

int main() {
    std::printf("streams benchmark — N=%d, chunk_size=%d, "
                "num_chunks=%d, total=%d, samples=%d\n",
                kN, kChunkSize, kNumChunks, kTotalMats, kSamples);

    // Build the base 12000-matrix host workload. We replicate this set
    // across kTotalMats device-side; chunk_idx % 6 selects which slice of
    // the base is used (12000 / chunk_size = 6 slices that exhaust the
    // 12000 matrices, then it cycles).
    constexpr int kBase = 12000;
    constexpr std::uint64_t kSeed = 0xD00D1E5701ULL;
    std::printf("generating %d random (H, S) pairs on host...\n", kBase);
    auto t_gen0 = std::chrono::steady_clock::now();
    const std::size_t mat = static_cast<std::size_t>(kN) * kN;
    std::vector<cdouble> H_base(mat * kBase);
    std::vector<cdouble> S_base(mat * kBase);
    {
        std::mt19937_64 rng(kSeed);
        for (int b = 0; b < kBase; ++b) {
            make_hermitian (kN, rng, &H_base[mat * b]);
            make_hpd_random(kN, rng, &S_base[mat * b]);
        }
    }
    auto t_gen1 = std::chrono::steady_clock::now();
    std::printf("host gen: %.1f s\n",
                std::chrono::duration<double>(t_gen1 - t_gen0).count());

    // Pick kSpotCount well-spread reference indices; precompute LAPACK
    // for those. Each device-side matrix at index `i` corresponds to
    // base index `i % kBase`, so we can recover the reference for any i.
    std::vector<int> spot_indices(kSpotCount);
    for (int k = 0; k < kSpotCount; ++k) {
        // Spread across kTotalMats; modulo kBase gives the base idx.
        spot_indices[k] = (k * (kTotalMats / kSpotCount)) % kTotalMats;
    }
    std::vector<LapackResult> spot_ref(kSpotCount);
    {
        std::printf("computing %d LAPACK reference solves...\n", kSpotCount);
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < kSpotCount; ++k) {
            const int base_idx = spot_indices[k] % kBase;
            spot_ref[k] = lapack_zhegv(&H_base[mat * base_idx],
                                       &S_base[mat * base_idx], kN);
        }
        auto t1 = std::chrono::steady_clock::now();
        std::printf("LAPACK ref: %.2f s\n",
                    std::chrono::duration<double>(t1 - t0).count());
    }

    UnifiedGeneigSolver solver(kN, /*device_id=*/0, /*bpb=*/1);
    std::printf("solver: BPB=%d, BlockDim.x=%u, shared_mem=%u\n",
                solver.batches_per_block(), solver.block_dim().x,
                solver.shared_mem_bytes());

    // Allocate device buffers sized for the full kTotalMats workload.
    const std::size_t mat_bytes_total = mat * kTotalMats * sizeof(cuDoubleComplex);
    const std::size_t w_bytes_total   = static_cast<std::size_t>(kN) * kTotalMats * sizeof(double);
    const std::size_t info_bytes_total= static_cast<std::size_t>(kTotalMats) * sizeof(int);
    std::printf("device allocation: H+S+U=%.2f GB, W=%.0f MB, info=%.0f MB\n",
                3.0 * mat_bytes_total / 1e9,
                w_bytes_total / 1e6, info_bytes_total / 1e6);

    CUdeviceptr d_H = 0, d_S = 0, d_U = 0, d_W = 0, d_info = 0;
    CU_CHECK(cuMemAlloc(&d_H,    mat_bytes_total));
    CU_CHECK(cuMemAlloc(&d_S,    mat_bytes_total));
    CU_CHECK(cuMemAlloc(&d_U,    mat_bytes_total));
    CU_CHECK(cuMemAlloc(&d_W,    w_bytes_total));
    CU_CHECK(cuMemAlloc(&d_info, info_bytes_total));

    // Tile the kBase matrices into the kTotalMats device slots in
    // segments of kBase. kTotalMats / kBase = 5 (60000 / 12000), and
    // (kTotalMats % kBase) == 0 so every device slot is filled by a
    // direct copy of the base buffer.
    static_assert(kTotalMats % kBase == 0,
                  "test assumes kTotalMats is a multiple of kBase");
    const std::size_t base_bytes = mat * kBase * sizeof(cuDoubleComplex);
    for (int rep = 0; rep < kTotalMats / kBase; ++rep) {
        const std::size_t off = static_cast<std::size_t>(rep) * base_bytes;
        CU_CHECK(cuMemcpyHtoD(d_H + off, H_base.data(), base_bytes));
        CU_CHECK(cuMemcpyHtoD(d_S + off, S_base.data(), base_bytes));
    }
    CU_CHECK(cuMemsetD8(d_info, 0, info_bytes_total));

    // Warm-up: one chunk on the default stream.
    solver.launch(d_H, d_S, d_W, d_U, d_info, kChunkSize, /*stream=*/nullptr);
    CU_CHECK(cuCtxSynchronize());

    auto run_one = [&](int num_streams) -> Sample {
        std::vector<CUstream> streams = make_streams(num_streams);
        // Reset d_info each run so failures don't leak across runs.
        CU_CHECK(cuMemsetD8(d_info, 0, info_bytes_total));

        CUevent e0, e1;
        cuEventCreate(&e0, CU_EVENT_DEFAULT);
        cuEventCreate(&e1, CU_EVENT_DEFAULT);
        // Record start on the first stream; record end after all streams
        // finish via cuStreamWaitEvent + a join event.
        cuEventRecord(e0, /*stream=*/nullptr);
        // Make every stream wait on the start event so they all begin
        // after this point (matters for accurate timing).
        for (CUstream s : streams) {
            cuStreamWaitEvent(s, e0, 0);
        }
        solver.launch_chunked(d_H, d_S, d_W, d_U, d_info,
                              kTotalMats, kChunkSize, num_streams, streams);
        // Record end after every stream completes.
        std::vector<CUevent> ends(num_streams);
        for (int i = 0; i < num_streams; ++i) {
            cuEventCreate(&ends[i], CU_EVENT_DEFAULT);
            cuEventRecord(ends[i], streams[i]);
        }
        // Default stream waits on all stream-end events, then we record e1.
        for (CUevent ev : ends) {
            cuStreamWaitEvent(/*stream=*/nullptr, ev, 0);
        }
        cuEventRecord(e1, /*stream=*/nullptr);
        CU_CHECK(cuCtxSynchronize());

        float ms = 0.f;
        cuEventElapsedTime(&ms, e0, e1);
        for (CUevent ev : ends) cuEventDestroy(ev);
        cuEventDestroy(e0); cuEventDestroy(e1);

        Sample r;
        r.mps = static_cast<double>(kTotalMats) / (ms / 1000.0);

        // Spot-check correctness on kSpotCount matrices.
        std::vector<cdouble> U_g(mat);
        std::vector<double>  W_g(kN);
        std::vector<cdouble> Sv(kN);
        for (int k = 0; k < kSpotCount; ++k) {
            const int       idx       = spot_indices[k];
            const int       base_idx  = idx % kBase;
            const cdouble*  S_b       = &S_base[mat * base_idx];
            const cdouble*  U_ref     = spot_ref[k].U.data();
            const double*   W_ref     = spot_ref[k].W.data();

            CU_CHECK(cuMemcpyDtoH(U_g.data(),
                                  d_U + static_cast<std::size_t>(idx) * mat * sizeof(cuDoubleComplex),
                                  mat * sizeof(cuDoubleComplex)));
            CU_CHECK(cuMemcpyDtoH(W_g.data(),
                                  d_W + static_cast<std::size_t>(idx) * kN * sizeof(double),
                                  kN * sizeof(double)));

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
                for (int j = 0; j < kN; ++j) inner += std::conj(u_g[j]) * Sv[j];
                const double phase_err = 1.0 - std::abs(inner);
                const double gap_left  = (i > 0)        ? std::abs(W_ref[i] - W_ref[i - 1]) : INFINITY;
                const double gap_right = (i < kN - 1)   ? std::abs(W_ref[i + 1] - W_ref[i]) : INFINITY;
                const double gap = std::min(gap_left, gap_right);
                if (gap >= kDegenGap) m_phase = std::max(m_phase, phase_err);
            }
            r.max_eig_rel = std::max(r.max_eig_rel, m_eig);
            r.max_phase   = std::max(r.max_phase,   m_phase);
        }
        destroy_streams(streams);
        return r;
    };

    const std::vector<int> stream_configs = {1, 2, 4, 8};
    std::vector<ConfigResult> results;
    for (int ns : stream_configs) {
        ConfigResult cr; cr.num_streams = ns;
        std::printf("\n--- %d-stream config ---\n", ns);
        for (int s = 0; s < kSamples; ++s) {
            Sample smp = run_one(ns);
            cr.mps_samples.push_back(smp.mps);
            cr.max_eig_rel = std::max(cr.max_eig_rel, smp.max_eig_rel);
            cr.max_phase   = std::max(cr.max_phase,   smp.max_phase);
            std::printf("  sample %d: %.0f mat/s  (eig_rel=%.2e, phase=%.2e)\n",
                        s + 1, smp.mps, smp.max_eig_rel, smp.max_phase);
        }
        cr.median_mps = median(cr.mps_samples);
        cr.min_mps    = *std::min_element(cr.mps_samples.begin(), cr.mps_samples.end());
        cr.max_mps    = *std::max_element(cr.mps_samples.begin(), cr.mps_samples.end());
        cr.pass       = (cr.max_eig_rel < kTolEig) && (cr.max_phase < kTolVec);
        results.push_back(std::move(cr));
    }

    // Result table.
    std::printf("\n");
    std::printf("================ streams benchmark (N=%d, chunk=%d, total=%d) ================\n",
                kN, kChunkSize, kTotalMats);
    std::printf("%-14s | %14s | %22s | %14s | %s\n",
                "stream config", "median mat/sec", "range (min..max)",
                "speedup vs 1S", "correctness");
    std::printf("---------------+----------------+------------------------+----------------+-----------\n");
    const double base_med = results.front().median_mps;
    for (const auto& cr : results) {
        char range_buf[64];
        std::snprintf(range_buf, sizeof(range_buf),
                      "%.0f .. %.0f", cr.min_mps, cr.max_mps);
        char spd_buf[32];
        std::snprintf(spd_buf, sizeof(spd_buf), "%.3fx", cr.median_mps / base_med);
        std::printf("%2d-stream      | %14.0f | %22s | %14s | %s "
                    "(eig_rel=%.2e, phase=%.2e)\n",
                    cr.num_streams, cr.median_mps, range_buf, spd_buf,
                    cr.pass ? "PASS" : "FAIL",
                    cr.max_eig_rel, cr.max_phase);
    }
    std::printf("============================================================================================\n");

    CU_CHECK(cuMemFree(d_H));
    CU_CHECK(cuMemFree(d_S));
    CU_CHECK(cuMemFree(d_U));
    CU_CHECK(cuMemFree(d_W));
    CU_CHECK(cuMemFree(d_info));

    bool all_pass = true;
    for (const auto& cr : results) all_pass = all_pass && cr.pass;
    return all_pass ? 0 : 1;
}
