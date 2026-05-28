// reduce-only kernel source for the hybrid path.
//
// This NVRTC source compiles a kernel that runs the Cholesky reduction
// stage of the generalized eigenvalue pipeline:
//   potrf(S) → L          (via cuSolverDx::Cholesky)
//   M := L⁻¹ · H · L⁻ᴴ    (via cuSolverDx::Trsm × 2)
// Outputs L and M to global memory. The HybridGeneigSolver then dispatches
// cusolverDnXsyevBatched on M and cublasZtrsmBatched on (L, V) for the
// remaining heev + back-transform stages — see hybrid_solver.cpp.
//
// Macros filled in at NVRTC compile time (same conventions as the unified
// kernel, see ../unified/kernel_source.hpp):
//   M_SIZE, SOLVER_LDA, SOLVER_SM, BATCHES_PER_BLOCK, BLOCK_DIM_X.
//
// Kernel exposes the same `__constant__` introspection symbols as the
// unified kernel:
//   solver_block_dim, solver_shared_memory_size, solver_batches_per_block.
//
// Kernel entry: `geneig_reduce_kernel`. Note this differs from the unified
// kernel's `geneig_full_kernel` so that a single C++ translation unit can
// link both without symbol collision (used in test_heev_fraction.cpp).

#pragma once

namespace sunny_geneig {

inline constexpr const char* kReduceKernelSource = R"kernel(
#include <cusolverdx.hpp>

using namespace cusolverdx;

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

using DType = typename Cholesky::a_data_type;

constexpr int          kN_   = M_SIZE;
constexpr int          kLDA_ = SOLVER_LDA;
constexpr unsigned int kBPB_ = BATCHES_PER_BLOCK;

// Reduce-only smem: As (BPB·N·LDA cdouble) + Bs (BPB·N·LDA cdouble).
constexpr unsigned int kReduceSmem = 2u * sizeof(DType) * kN_ * kLDA_ * kBPB_;

__constant__ dim3         solver_block_dim          = Cholesky::block_dim;
__constant__ unsigned int solver_shared_memory_size = kReduceSmem;
__constant__ unsigned int solver_batches_per_block  = kBPB_;

extern "C" __global__ __launch_bounds__(Cholesky::max_threads_per_block)
void geneig_reduce_kernel(const cuDoubleComplex* __restrict__ H_in,
                          const cuDoubleComplex* __restrict__ S_in,
                          cuDoubleComplex*       __restrict__ L_out,
                          cuDoubleComplex*       __restrict__ M_out,
                          int*                   __restrict__ info_out,
                          int                                 batch_size) {
    const int batch_idx = blockIdx.x * kBPB_;
    if (batch_idx >= batch_size) return;

    const unsigned long long mat_stride = (unsigned long long)kN_ * kLDA_;

    const cuDoubleComplex* H_b   = H_in   + mat_stride * batch_idx;
    const cuDoubleComplex* S_b   = S_in   + mat_stride * batch_idx;
    cuDoubleComplex*       L_b   = L_out  + mat_stride * batch_idx;
    cuDoubleComplex*       M_b   = M_out  + mat_stride * batch_idx;
    int*                   info_b = info_out + batch_idx;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr unsigned int kBsBytes = sizeof(DType) * kN_ * kLDA_ * kBPB_;

    DType* As = reinterpret_cast<DType*>(smem_raw);
    DType* Bs = reinterpret_cast<DType*>(smem_raw + kBsBytes);

    const DType* S_typed = reinterpret_cast<const DType*>(S_b);
    const DType* H_typed = reinterpret_cast<const DType*>(H_b);
    const int total_elems = kN_ * kLDA_ * (int)kBPB_;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    Cholesky().execute(As, info_b);
    __syncthreads();
    TrsmLeft().execute(As, kLDA_, Bs, kLDA_);
    __syncthreads();
    TrsmRight().execute(As, kLDA_, Bs, kLDA_);
    __syncthreads();

    // Writeback: L (the Cholesky factor in As) and M = L⁻¹ H L⁻ᴴ (in Bs).
    DType* L_typed = reinterpret_cast<DType*>(L_b);
    DType* M_typed = reinterpret_cast<DType*>(M_b);
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        L_typed[idx] = As[idx];
        M_typed[idx] = Bs[idx];
    }
}
)kernel";

}  // namespace sunny_geneig
