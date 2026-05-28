// unified five-stage NVRTC kernel with BatchesPerBlock support.
//
// Each block processes BATCHES_PER_BLOCK matrices contiguously, mirroring the
// canonical cuSolverDx-shipped batched kernel pattern (see
// example/cusolverdx/04_Symmetric_Eigenvalues/heev_batched.cu and
// example/cusolverdx/00_Introduction/posv_batched.cu).
//
// Macros filled in at NVRTC compile time:
//   M_SIZE             — matrix dimension (driven by UnifiedGeneigSolver::n)
//   SOLVER_LDA         — leading dimension (= M_SIZE)
//   SOLVER_SM          — cuSolverDx tuning-database tag (e.g. 800)
//   BATCHES_PER_BLOCK  — number of matrices each block solves (>= 1)
//
// Caller contract: `batch_size` must be a multiple of BATCHES_PER_BLOCK
// (the launch wrapper computes `gridDim.x = batch_size / BATCHES_PER_BLOCK`
// and the kernel does NOT bounds-check the partial last block — pad your
// device buffers to padded_batches if needed).
//
// Kernel exposes:
//   solver_block_dim          (dim3)            — `__constant__`
//   solver_shared_memory_size (unsigned int)    — `__constant__`
//   solver_batches_per_block  (unsigned int)    — `__constant__`
//
// Kernel entry: `geneig_full_kernel`.
//
// The hybrid path's reduce-only kernel source lives in
// ../hybrid/reduce_kernel_source.hpp (split out in to keep
// unified and hybrid code paths independent).

#pragma once

namespace sunny_geneig {

// ---------------------------------------------------------------------------
// Probe kernel: defines the same five solver types but exposes only their
// suggested_block_dim values via __constant__. No execute() calls; the empty
// `__global__ probe_kernel` is a hook for nvJitLink. NVRTC + nvJitLink + load
// completes in roughly 5 s on the development hardware (vs ~228 s for the
// full kernel), so we use this for cheap pre-flight to discover the per-
// operator block-dim recommendation before the production compile.
// ---------------------------------------------------------------------------
inline constexpr const char* kProbeSource = R"probe(
#include <cusolverdx.hpp>
using namespace cusolverdx;

using Cholesky = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>() + Type<type::complex>()
                        + Function<function::potrf>() + FillMode<lower>()
                        + Arrangement<col_major>() + LeadingDimension<SOLVER_LDA>()
                        + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                        + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmLeft = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>() + Type<type::complex>()
                        + Function<function::trsm>() + Side<side::left>()
                        + FillMode<lower>() + TransposeMode<non_trans>()
                        + Diag<diag::non_unit>()
                        + Arrangement<col_major, col_major>()
                        + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                        + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                        + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmRight = decltype(Size<M_SIZE, M_SIZE>()
                         + Precision<double>() + Type<type::complex>()
                         + Function<function::trsm>() + Side<side::right>()
                         + FillMode<lower>() + TransposeMode<conj_trans>()
                         + Diag<diag::non_unit>()
                         + Arrangement<col_major, col_major>()
                         + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                         + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                         + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmLeftConj = decltype(Size<M_SIZE, M_SIZE>()
                            + Precision<double>() + Type<type::complex>()
                            + Function<function::trsm>() + Side<side::left>()
                            + FillMode<lower>() + TransposeMode<conj_trans>()
                            + Diag<diag::non_unit>()
                            + Arrangement<col_major, col_major>()
                            + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                            + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                            + BatchesPerBlock<BATCHES_PER_BLOCK>());

using Heev = decltype(Size<M_SIZE>()
                    + Precision<double>() + Type<type::complex>()
                    + Function<function::heev>() + FillMode<lower>()
                    + Arrangement<col_major>() + LeadingDimension<SOLVER_LDA>()
                    + Job<job::overwrite_vectors>()
                    + SM<SOLVER_SM>() + Block() + BlockDim<128>()
                    + BatchesPerBlock<BATCHES_PER_BLOCK>());

__constant__ dim3 cholesky_suggested_block_dim     = Cholesky::suggested_block_dim;
__constant__ dim3 trsm_left_suggested_block_dim    = TrsmLeft::suggested_block_dim;
__constant__ dim3 trsm_right_suggested_block_dim   = TrsmRight::suggested_block_dim;
__constant__ dim3 trsm_lc_suggested_block_dim      = TrsmLeftConj::suggested_block_dim;
__constant__ dim3 heev_suggested_block_dim         = Heev::suggested_block_dim;

extern "C" __global__ void probe_kernel() {}
)probe";

inline constexpr const char* kKernelSource = R"kernel(
#include <cusolverdx.hpp>

using namespace cusolverdx;

// Operator-ordering note: under NVRTC, SM<> must come BEFORE Block() in the
// description. See cusolverdx/detail/solver_description_arithmetic.hpp.

using Cholesky = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>()
                        + Type<type::complex>()
                        + Function<function::potrf>()
                        + FillMode<lower>()
                        + Arrangement<col_major>()
                        + LeadingDimension<SOLVER_LDA>()
                        + SM<SOLVER_SM>()
                        + Block()
                        + BlockDim<BLOCK_DIM_X>()
                        + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmLeft = decltype(Size<M_SIZE, M_SIZE>()
                        + Precision<double>()
                        + Type<type::complex>()
                        + Function<function::trsm>()
                        + Side<side::left>()
                        + FillMode<lower>()
                        + TransposeMode<non_trans>()
                        + Diag<diag::non_unit>()
                        + Arrangement<col_major, col_major>()
                        + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                        + SM<SOLVER_SM>()
                        + Block()
                        + BlockDim<BLOCK_DIM_X>()
                        + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmRight = decltype(Size<M_SIZE, M_SIZE>()
                         + Precision<double>()
                         + Type<type::complex>()
                         + Function<function::trsm>()
                         + Side<side::right>()
                         + FillMode<lower>()
                         + TransposeMode<conj_trans>()
                         + Diag<diag::non_unit>()
                         + Arrangement<col_major, col_major>()
                         + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                         + SM<SOLVER_SM>()
                         + Block()
                         + BlockDim<BLOCK_DIM_X>()
                         + BatchesPerBlock<BATCHES_PER_BLOCK>());

using TrsmLeftConj = decltype(Size<M_SIZE, M_SIZE>()
                            + Precision<double>()
                            + Type<type::complex>()
                            + Function<function::trsm>()
                            + Side<side::left>()
                            + FillMode<lower>()
                            + TransposeMode<conj_trans>()
                            + Diag<diag::non_unit>()
                            + Arrangement<col_major, col_major>()
                            + LeadingDimension<SOLVER_LDA, SOLVER_LDA>()
                            + SM<SOLVER_SM>()
                            + Block()
                            + BlockDim<BLOCK_DIM_X>()
                            + BatchesPerBlock<BATCHES_PER_BLOCK>());

using Heev = decltype(Size<M_SIZE>()
                    + Precision<double>()
                    + Type<type::complex>()
                    + Function<function::heev>()
                    + FillMode<lower>()
                    + Arrangement<col_major>()
                    + LeadingDimension<SOLVER_LDA>()
                    + Job<job::overwrite_vectors>()
                    + SM<SOLVER_SM>()
                    + Block()
                    + BlockDim<BLOCK_DIM_X>()
                    + BatchesPerBlock<BATCHES_PER_BLOCK>());

using DType      = typename Cholesky::a_data_type;
using StatusType = typename Cholesky::status_type;
using PType      = typename Heev::a_precision;

constexpr int          kN              = M_SIZE;
constexpr int          kLDA            = SOLVER_LDA;
constexpr unsigned int kBPB            = BATCHES_PER_BLOCK;
// Cholesky::shared_memory_size and Heev::shared_memory_size already include
// the BatchesPerBlock factor (each tile is sized BPB * one_batch_size).
constexpr unsigned int kFullSmem =
    Cholesky::shared_memory_size + Heev::shared_memory_size;

__constant__ dim3         solver_block_dim          = Cholesky::block_dim;
__constant__ unsigned int solver_shared_memory_size = kFullSmem;
__constant__ unsigned int solver_batches_per_block  = kBPB;

extern "C" __global__ __launch_bounds__(Cholesky::max_threads_per_block)
void geneig_full_kernel(const cuDoubleComplex* __restrict__ H_in,
                        const cuDoubleComplex* __restrict__ S_in,
                        double*                __restrict__ W_out,
                        cuDoubleComplex*       __restrict__ U_out,
                        int*                   __restrict__ info_out,
                        int                                 batch_size) {
    const int batch_idx = blockIdx.x * kBPB;
    if (batch_idx >= batch_size) return;

    const unsigned long long mat_stride = (unsigned long long)kN * kLDA;
    const unsigned long long w_stride   = (unsigned long long)kN;

    const cuDoubleComplex* H_b   = H_in   + mat_stride * batch_idx;
    const cuDoubleComplex* S_b   = S_in   + mat_stride * batch_idx;
    cuDoubleComplex*       U_b   = U_out  + mat_stride * batch_idx;
    double*                W_b   = W_out  + w_stride   * batch_idx;
    int*                   info_b = info_out + batch_idx;

    extern __shared__ __align__(16) unsigned char smem_raw[];

    // Layout (BPB matrices contiguous per region):
    //   [0 .. Cholesky::shared_memory_size)
    //        As — BPB × (N×LDA) cdouble; S→L tiles stacked.
    //   [Cholesky::shared_memory_size .. + Heev::shared_memory_size)
    //        Bs (BPB × N×LDA cdouble), lambda_s (BPB × N real),
    //        workspace_s (heev internal scratch).
    constexpr unsigned int kBsOffset       = Cholesky::shared_memory_size;
    constexpr unsigned int kBsBytes        = sizeof(DType) * kN * kLDA * kBPB;
    constexpr unsigned int kLambdaOffset   = kBsOffset + kBsBytes;
    constexpr unsigned int kLambdaBytes    = sizeof(PType) * kN * kBPB;
    constexpr unsigned int kWorkspaceOffset =
        ((kLambdaOffset + kLambdaBytes) + alignof(DType) - 1) & ~(alignof(DType) - 1);

    DType* As           = reinterpret_cast<DType*>(smem_raw);
    DType* Bs           = reinterpret_cast<DType*>(smem_raw + kBsOffset);
    PType* lambda_s     = reinterpret_cast<PType*>(smem_raw + kLambdaOffset);
    DType* workspace_s  = reinterpret_cast<DType*>(smem_raw + kWorkspaceOffset);

    // Cooperative load: copy BPB matrices' worth of data, contiguous in both
    // global and shared memory (no padding since LDA == N).
    const DType* S_typed = reinterpret_cast<const DType*>(S_b);
    const DType* H_typed = reinterpret_cast<const DType*>(H_b);
    const int total_elems = kN * kLDA * (int)kBPB;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    Cholesky().execute(As, info_b);
    __syncthreads();
    TrsmLeft().execute(As, kLDA, Bs, kLDA);
    __syncthreads();
    TrsmRight().execute(As, kLDA, Bs, kLDA);
    __syncthreads();
    Heev().execute(Bs, kLDA, lambda_s, workspace_s, info_b);
    __syncthreads();
    TrsmLeftConj().execute(As, kLDA, Bs, kLDA);
    __syncthreads();
    // [3.5c experiment] Removing the post-Cholesky and post-Heev syncs is
    // *defensible from cuSolverDx source* (their dispatch ends with
    // __syncthreads()), but empirically REGRESSES throughput by ~14% on
    // N=54 BPB=1 on sm_120 (5070 Ti laptop). Five-sample medians:
    //   3.5b (5 syncs): 6,740 mat/s
    //   3.5c (3 syncs): 5,791 mat/s   ← regression
    // The most likely explanation is that cuSolverDx's internal sync syncs
    // only a subset of the warps (per the partial-warp/warp-per-batch
    // dispatch heuristics in cholesky.cuh), so the external sync we
    // removed was actually doing useful work for downstream operators that
    // span the full CTA. The kernel keeps all 5 syncs for correctness AND
    // throughput. See docs/NVRTC_RUNTIME.md "3.5c experiment" for the
    // detailed measurement record.

    // Writeback: BPB lambda blocks (each kN reals) and BPB U tiles (each kN×kLDA cdouble).
    const int total_w = kN * (int)kBPB;
    for (int idx = threadIdx.x; idx < total_w; idx += blockDim.x) {
        W_b[idx] = lambda_s[idx];
    }
    DType* U_typed = reinterpret_cast<DType*>(U_b);
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        U_typed[idx] = Bs[idx];
    }
}
)kernel";

}  // namespace sunny_geneig

