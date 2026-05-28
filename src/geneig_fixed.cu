// Step 2a — fixed-size N=54 Cholesky kernel using cuSolverDx.
//
// Single-block, single-matrix in-kernel Cholesky factorisation of a 54x54
// complex Hermitian positive-definite matrix S, producing the lower triangular
// factor L. Strict upper triangle of L_out is explicitly zeroed for clean
// element-wise comparison against a host LAPACK reference.

#include <cusolverdx.hpp>

#include <cuComplex.h>
#include <cuda_runtime.h>

namespace {

using namespace cusolverdx;

using Cholesky54 = decltype(Size<54, 54>()
                          + Precision<double>()
                          + Type<type::complex>()
                          + Function<potrf>()
                          + FillMode<fill_mode::lower>()
                          + Arrangement<arrangement::col_major>()
                          + LeadingDimension<54>()
                          + Block()
                          + BlockDim<128>()
                          + SM<800>());

// trsm operands: A is the triangular factor L (54x54, lower, non-unit diag),
// B is H/Y/M (54x54). Both column-major, no padding.
//
// Trsm54Left  — solves L * Y = H        (Side=left,  TransposeMode=non_trans)  → Bs ← L^{-1} * Bs
// Trsm54Right — solves Y * L^H = M      (Side=right, TransposeMode=conj_trans) → Bs ← Bs * L^{-H}
using Trsm54Left = decltype(Size<54, 54>()
                          + Precision<double>()
                          + Type<type::complex>()
                          + Function<function::trsm>()
                          + Side<side::left>()
                          + FillMode<lower>()
                          + TransposeMode<non_trans>()
                          + Diag<diag::non_unit>()
                          + Arrangement<col_major, col_major>()
                          + LeadingDimension<54, 54>()
                          + Block()
                          + BlockDim<128>()
                          + SM<800>());

using Trsm54Right = decltype(Size<54, 54>()
                           + Precision<double>()
                           + Type<type::complex>()
                           + Function<function::trsm>()
                           + Side<side::right>()
                           + FillMode<lower>()
                           + TransposeMode<conj_trans>()
                           + Diag<diag::non_unit>()
                           + Arrangement<col_major, col_major>()
                           + LeadingDimension<54, 54>()
                           + Block()
                           + BlockDim<128>()
                           + SM<800>());

// Back-transform: solve L^H * U = V for U → U = L^{-H} V (eigenvectors of the
// generalised problem). Same triangular factor L still resident in As.
using Trsm54LeftConj = decltype(Size<54, 54>()
                              + Precision<double>()
                              + Type<type::complex>()
                              + Function<function::trsm>()
                              + Side<side::left>()
                              + FillMode<lower>()
                              + TransposeMode<conj_trans>()
                              + Diag<diag::non_unit>()
                              + Arrangement<col_major, col_major>()
                              + LeadingDimension<54, 54>()
                              + Block()
                              + BlockDim<128>()
                              + SM<800>());

// Heev54 — Hermitian eigendecomposition of M (54x54), in-place: Bs ← V on
// success, with eigenvalues in lambda_s. Algorithm is the cuSolverDx QR-based
// path (function::heev). Job<job::overwrite_vectors>() is required for
// eigenvectors.
using Heev54 = decltype(Size<54>()
                      + Precision<double>()
                      + Type<type::complex>()
                      + Function<function::heev>()
                      + FillMode<lower>()
                      + Arrangement<col_major>()
                      + LeadingDimension<54>()
                      + Job<job::overwrite_vectors>()
                      + Block()
                      + BlockDim<128>()
                      + SM<800>());

using DType      = typename Cholesky54::a_data_type;
using StatusType = typename Cholesky54::status_type;
using PType      = typename Heev54::a_precision;

constexpr int kN   = 54;
constexpr int kLDA = 54;
constexpr unsigned int kReduceSmem =
    Cholesky54::shared_memory_size + Cholesky54::shared_memory_size;
constexpr unsigned int kReduceEigSmem =
    Cholesky54::shared_memory_size + Heev54::shared_memory_size;

}  // namespace

__global__ __launch_bounds__(Cholesky54::max_threads_per_block)
void chol_n54_kernel(const cuDoubleComplex* __restrict__ S_in,
                     cuDoubleComplex*       __restrict__ L_out,
                     int*                                info) {
    extern __shared__ unsigned char smem_raw[];
    DType* As = reinterpret_cast<DType*>(smem_raw);

    const DType* S_typed = reinterpret_cast<const DType*>(S_in);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        As[idx] = S_typed[idx];
    }
    __syncthreads();

    Cholesky54().execute(As, reinterpret_cast<StatusType*>(info));
    __syncthreads();

    DType* L_typed = reinterpret_cast<DType*>(L_out);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        int col = idx / kLDA;
        int row = idx % kLDA;
        if (row >= col) {
            L_typed[idx] = As[idx];
        } else {
            L_typed[idx] = DType{};
        }
    }
}

extern "C" void chol_n54_launch(const cuDoubleComplex* d_S,
                                cuDoubleComplex*       d_L,
                                int*                   d_info,
                                cudaStream_t           stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(chol_n54_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Cholesky54::shared_memory_size);
        attr_set = true;
    }
    chol_n54_kernel<<<1, Cholesky54::block_dim,
                      Cholesky54::shared_memory_size, stream>>>(d_S, d_L, d_info);
}

extern "C" {
unsigned int chol_n54_block_dim_x()       { return Cholesky54::block_dim.x; }
unsigned int chol_n54_shared_mem_bytes()  { return Cholesky54::shared_memory_size; }
}

// ===========================================================================
// Step 2b — full reduction kernel: M = L^{-1} H L^{-H}.
//
// Layout in dynamic shared memory:
//   [0 .. 46656)  : As — holds S on entry, transformed in place to the
//                   Cholesky factor L by Cholesky54.
//   [46656 .. 93312) : Bs — holds H on entry, transformed in place by the
//                   two trsm calls into the reduced Hermitian matrix M.
// Both tiles are column-major with leading dimension 54, contiguous in shared
// memory; no padding.
// ===========================================================================

__global__ __launch_bounds__(Cholesky54::max_threads_per_block)
void geneig_reduce_n54_kernel(const cuDoubleComplex* __restrict__ H_in,
                              const cuDoubleComplex* __restrict__ S_in,
                              cuDoubleComplex*       __restrict__ M_out,
                              int*                                info_out) {
    extern __shared__ unsigned char smem_raw[];
    DType* As = reinterpret_cast<DType*>(smem_raw);
    DType* Bs = reinterpret_cast<DType*>(smem_raw + Cholesky54::shared_memory_size);

    const DType* S_typed = reinterpret_cast<const DType*>(S_in);
    const DType* H_typed = reinterpret_cast<const DType*>(H_in);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    // S = L L^H — overwrites As lower triangle with L.
    Cholesky54().execute(As, reinterpret_cast<StatusType*>(info_out));
    __syncthreads();

    // Bs ← L^{-1} * Bs   (left-multiply by L^{-1}).
    Trsm54Left().execute(As, kLDA, Bs, kLDA);
    __syncthreads();

    // Bs ← Bs * L^{-H}   (right-multiply by L^{-H}).
    Trsm54Right().execute(As, kLDA, Bs, kLDA);
    __syncthreads();

    // Writeback: lower triangle of Bs is M; zero strict upper.
    DType* M_typed = reinterpret_cast<DType*>(M_out);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        int col = idx / kLDA;
        int row = idx % kLDA;
        if (row >= col) {
            M_typed[idx] = Bs[idx];
        } else {
            M_typed[idx] = DType{};
        }
    }
}

extern "C" void geneig_reduce_n54_launch(const cuDoubleComplex* d_H,
                                         const cuDoubleComplex* d_S,
                                         cuDoubleComplex*       d_M,
                                         int*                   d_info,
                                         cudaStream_t           stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(geneig_reduce_n54_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kReduceSmem);
        attr_set = true;
    }
    geneig_reduce_n54_kernel<<<1, Cholesky54::block_dim, kReduceSmem, stream>>>(
        d_H, d_S, d_M, d_info);
}

extern "C" {
unsigned int geneig_reduce_n54_block_dim_x()      { return Cholesky54::block_dim.x; }
unsigned int geneig_reduce_n54_shared_mem_bytes() { return kReduceSmem; }
}

// ===========================================================================
// Step 2c — full reduction + Hermitian eigendecomposition.
//
// Pipeline: S = L L^H ; M = L^{-1} H L^{-H} ; M V = V diag(W).
// Outputs: W (real eigenvalues, ascending) and V (eigenvectors of M; columns
// of Bs after heev). NOT yet back-transformed to eigenvectors of the original
// generalised problem — that is Step 2d.
//
// Shared-memory layout:
//   [0 .. Cholesky54::shared_memory_size)
//        As — S/L tile (kept live through both trsm calls; ignored by heev).
//   [Cholesky54::shared_memory_size .. + Heev54::shared_memory_size)
//        sliced via cusolverdx::shared_memory::slice into:
//          Bs       — H/M/V tile (54x54 cdouble)
//          lambda_s — real eigenvalues (54 doubles)
//          workspace_s — heev internal scratch
// ===========================================================================

__global__ __launch_bounds__(Cholesky54::max_threads_per_block)
void geneig_reduce_eig_n54_kernel(const cuDoubleComplex* __restrict__ H_in,
                                  const cuDoubleComplex* __restrict__ S_in,
                                  double*                __restrict__ W_out,
                                  cuDoubleComplex*       __restrict__ V_out,
                                  int*                                info_out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];

    // Manual offset arithmetic (avoids cute::tuple structured binding in
    // device code). Layout:
    //   [0 .. 46656)               As — S→L tile
    //   [46656 .. 93312)           Bs — H→M→V tile
    //   [93312 .. 93312+8*kN)      lambda_s — heev eigenvalues (real)
    //   [aligned up .. end)        workspace_s — heev internal scratch
    constexpr unsigned int kBsOffset       = Cholesky54::shared_memory_size;
    constexpr unsigned int kLambdaOffset   = kBsOffset + sizeof(DType) * kN * kLDA;
    constexpr unsigned int kLambdaBytes    = sizeof(PType) * kN;
    constexpr unsigned int kWorkspaceOffset =
        ((kLambdaOffset + kLambdaBytes) + alignof(DType) - 1) & ~(alignof(DType) - 1);

    DType* As           = reinterpret_cast<DType*>(smem_raw);
    DType* Bs           = reinterpret_cast<DType*>(smem_raw + kBsOffset);
    PType* lambda_s     = reinterpret_cast<PType*>(smem_raw + kLambdaOffset);
    DType* workspace_s  = reinterpret_cast<DType*>(smem_raw + kWorkspaceOffset);

    const DType* S_typed = reinterpret_cast<const DType*>(S_in);
    const DType* H_typed = reinterpret_cast<const DType*>(H_in);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    Cholesky54().execute(As, reinterpret_cast<StatusType*>(info_out));
    __syncthreads();

    Trsm54Left().execute(As, kLDA, Bs, kLDA);
    __syncthreads();

    Trsm54Right().execute(As, kLDA, Bs, kLDA);
    __syncthreads();

    Heev54().execute(Bs, kLDA, lambda_s, workspace_s,
                     reinterpret_cast<StatusType*>(info_out));
    __syncthreads();

    for (int i = threadIdx.x; i < kN; i += blockDim.x) {
        W_out[i] = lambda_s[i];
    }
    DType* V_typed = reinterpret_cast<DType*>(V_out);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        V_typed[idx] = Bs[idx];
    }
}

extern "C" void geneig_reduce_eig_n54_launch(const cuDoubleComplex* d_H,
                                             const cuDoubleComplex* d_S,
                                             double*                d_W,
                                             cuDoubleComplex*       d_V,
                                             int*                   d_info,
                                             cudaStream_t           stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(geneig_reduce_eig_n54_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kReduceEigSmem);
        attr_set = true;
    }
    geneig_reduce_eig_n54_kernel<<<1, Cholesky54::block_dim, kReduceEigSmem, stream>>>(
        d_H, d_S, d_W, d_V, d_info);
}

extern "C" {
unsigned int geneig_reduce_eig_n54_block_dim_x()       { return Cholesky54::block_dim.x; }
unsigned int geneig_reduce_eig_n54_shared_mem_bytes()  { return kReduceEigSmem; }
unsigned int geneig_reduce_eig_n54_chol_smem_bytes()   { return Cholesky54::shared_memory_size; }
unsigned int geneig_reduce_eig_n54_heev_smem_bytes()   { return Heev54::shared_memory_size; }
unsigned int geneig_reduce_eig_n54_heev_workspace()    { return Heev54::workspace_size; }
}

// ===========================================================================
// Step 2d — full generalised eigenvalue solver: H u = lambda S u.
//
// Pipeline (extends 2c with one more trsm):
//   S = L L^H ; M = L^{-1} H L^{-H} ; M V = V diag(W) ; U = L^{-H} V.
// Outputs: W (real eigenvalues, ascending) and U (generalised eigenvectors of
// the original problem, S-orthonormal: U^H S U = I).
//
// Shared-memory layout: identical to 2c. As (S/L) is kept resident through
// the whole pipeline so the back-transform trsm has L available.
// ===========================================================================

__global__ __launch_bounds__(Cholesky54::max_threads_per_block)
void geneig_full_n54_kernel(const cuDoubleComplex* __restrict__ H_in,
                            const cuDoubleComplex* __restrict__ S_in,
                            double*                __restrict__ W_out,
                            cuDoubleComplex*       __restrict__ U_out,
                            int*                                info_out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];

    constexpr unsigned int kBsOffset       = Cholesky54::shared_memory_size;
    constexpr unsigned int kLambdaOffset   = kBsOffset + sizeof(DType) * kN * kLDA;
    constexpr unsigned int kLambdaBytes    = sizeof(PType) * kN;
    constexpr unsigned int kWorkspaceOffset =
        ((kLambdaOffset + kLambdaBytes) + alignof(DType) - 1) & ~(alignof(DType) - 1);

    DType* As           = reinterpret_cast<DType*>(smem_raw);
    DType* Bs           = reinterpret_cast<DType*>(smem_raw + kBsOffset);
    PType* lambda_s     = reinterpret_cast<PType*>(smem_raw + kLambdaOffset);
    DType* workspace_s  = reinterpret_cast<DType*>(smem_raw + kWorkspaceOffset);

    const DType* S_typed = reinterpret_cast<const DType*>(S_in);
    const DType* H_typed = reinterpret_cast<const DType*>(H_in);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    Cholesky54().execute(As, reinterpret_cast<StatusType*>(info_out));
    __syncthreads();
    Trsm54Left().execute(As, kLDA, Bs, kLDA);            // Bs ← L^{-1} H
    __syncthreads();
    Trsm54Right().execute(As, kLDA, Bs, kLDA);           // Bs ← Bs L^{-H} = M
    __syncthreads();
    Heev54().execute(Bs, kLDA, lambda_s, workspace_s,
                     reinterpret_cast<StatusType*>(info_out));
    __syncthreads();                                      // Bs = V (eigvecs of M)
    Trsm54LeftConj().execute(As, kLDA, Bs, kLDA);        // Bs ← L^{-H} V = U
    __syncthreads();

    for (int i = threadIdx.x; i < kN; i += blockDim.x)
        W_out[i] = lambda_s[i];
    DType* U_typed = reinterpret_cast<DType*>(U_out);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x)
        U_typed[idx] = Bs[idx];
}

extern "C" void geneig_full_n54_launch(const cuDoubleComplex* d_H,
                                       const cuDoubleComplex* d_S,
                                       double*                d_W,
                                       cuDoubleComplex*       d_U,
                                       int*                   d_info,
                                       cudaStream_t           stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(geneig_full_n54_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kReduceEigSmem);
        attr_set = true;
    }
    geneig_full_n54_kernel<<<1, Cholesky54::block_dim, kReduceEigSmem, stream>>>(
        d_H, d_S, d_W, d_U, d_info);
}

extern "C" {
unsigned int geneig_full_n54_block_dim_x()      { return Cholesky54::block_dim.x; }
unsigned int geneig_full_n54_shared_mem_bytes() { return kReduceEigSmem; }
}

// ===========================================================================
// Step 2e — batched dispatch: one block per matrix.
//
// Mathematically identical to geneig_full_n54_kernel; the only difference is
// per-block pointer arithmetic on the global-memory inputs/outputs and an
// early-return guard for over-launched blocks.
// ===========================================================================

__global__ __launch_bounds__(Cholesky54::max_threads_per_block)
void geneig_full_n54_batched_kernel(const cuDoubleComplex* __restrict__ H_in,
                                    const cuDoubleComplex* __restrict__ S_in,
                                    double*                __restrict__ W_out,
                                    cuDoubleComplex*       __restrict__ U_out,
                                    int*                   __restrict__ info_out,
                                    int                                 batch_size) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const std::size_t mat_stride = static_cast<std::size_t>(kN) * kLDA;
    const std::size_t w_stride   = static_cast<std::size_t>(kN);

    const cuDoubleComplex* H_b = H_in + mat_stride * b;
    const cuDoubleComplex* S_b = S_in + mat_stride * b;
    cuDoubleComplex*       U_b = U_out + mat_stride * b;
    double*                W_b = W_out + w_stride   * b;
    int*                   info_b = info_out + b;

    extern __shared__ __align__(16) unsigned char smem_raw[];

    constexpr unsigned int kBsOffset       = Cholesky54::shared_memory_size;
    constexpr unsigned int kLambdaOffset   = kBsOffset + sizeof(DType) * kN * kLDA;
    constexpr unsigned int kLambdaBytes    = sizeof(PType) * kN;
    constexpr unsigned int kWorkspaceOffset =
        ((kLambdaOffset + kLambdaBytes) + alignof(DType) - 1) & ~(alignof(DType) - 1);

    DType* As           = reinterpret_cast<DType*>(smem_raw);
    DType* Bs           = reinterpret_cast<DType*>(smem_raw + kBsOffset);
    PType* lambda_s     = reinterpret_cast<PType*>(smem_raw + kLambdaOffset);
    DType* workspace_s  = reinterpret_cast<DType*>(smem_raw + kWorkspaceOffset);

    const DType* S_typed = reinterpret_cast<const DType*>(S_b);
    const DType* H_typed = reinterpret_cast<const DType*>(H_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        As[idx] = S_typed[idx];
        Bs[idx] = H_typed[idx];
    }
    __syncthreads();

    Cholesky54().execute(As, reinterpret_cast<StatusType*>(info_b));
    __syncthreads();
    Trsm54Left().execute(As, kLDA, Bs, kLDA);
    __syncthreads();
    Trsm54Right().execute(As, kLDA, Bs, kLDA);
    __syncthreads();
    Heev54().execute(Bs, kLDA, lambda_s, workspace_s,
                     reinterpret_cast<StatusType*>(info_b));
    __syncthreads();
    Trsm54LeftConj().execute(As, kLDA, Bs, kLDA);
    __syncthreads();

    for (int i = threadIdx.x; i < kN; i += blockDim.x)
        W_b[i] = lambda_s[i];
    DType* U_typed = reinterpret_cast<DType*>(U_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x)
        U_typed[idx] = Bs[idx];
}

extern "C" void geneig_full_n54_batched_launch(const cuDoubleComplex* d_H,
                                               const cuDoubleComplex* d_S,
                                               double*                d_W,
                                               cuDoubleComplex*       d_U,
                                               int*                   d_info,
                                               int                    batch_size,
                                               cudaStream_t           stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(geneig_full_n54_batched_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kReduceEigSmem);
        attr_set = true;
    }
    if (batch_size <= 0) return;
    geneig_full_n54_batched_kernel<<<batch_size, Cholesky54::block_dim,
                                     kReduceEigSmem, stream>>>(
        d_H, d_S, d_W, d_U, d_info, batch_size);
}

extern "C" {
unsigned int geneig_full_n54_batched_block_dim_x()      { return Cholesky54::block_dim.x; }
unsigned int geneig_full_n54_batched_shared_mem_bytes() { return kReduceEigSmem; }
}

// ===========================================================================
// finishing kernel for the hybrid NVRTC + static pipeline.
//
// Inputs (computed by the NVRTC `geneig_reduce_kernel`):
//   L_in : Cholesky factor L of S, lower triangular, strict upper zeroed.
//   M_in : reduced matrix L^{-1} H L^{-H} (full Hermitian; only lower
//          triangle is read by heev under FillMode<lower>).
// Outputs:
//   W_out : real eigenvalues of M (= eigenvalues of the generalised problem).
//   U_out : generalised eigenvectors u_i (S-orthonormal).
//
// Pipeline: heev(M) → V; trsm_left_conj(L, V) → U = L^{-H} V.
// Shared-memory layout matches geneig_reduce_eig_n54_kernel: As holds L
// throughout (needed by the back-transform trsm), Bs holds M → V → U, with
// the heev workspace and lambda_s tucked into Heev54::shared_memory_size.
// ===========================================================================

__global__ __launch_bounds__(Cholesky54::max_threads_per_block)
void geneig_finish_n54_kernel(const cuDoubleComplex* __restrict__ L_in,
                              const cuDoubleComplex* __restrict__ M_in,
                              double*                __restrict__ W_out,
                              cuDoubleComplex*       __restrict__ U_out,
                              int*                   __restrict__ info_out,
                              int                                 batch_size) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const std::size_t mat_stride = static_cast<std::size_t>(kN) * kLDA;
    const std::size_t w_stride   = static_cast<std::size_t>(kN);

    const cuDoubleComplex* L_b = L_in  + mat_stride * b;
    const cuDoubleComplex* M_b = M_in  + mat_stride * b;
    cuDoubleComplex*       U_b = U_out + mat_stride * b;
    double*                W_b = W_out + w_stride   * b;
    int*                   info_b = info_out + b;

    extern __shared__ __align__(16) unsigned char smem_raw[];

    constexpr unsigned int kBsOffset       = Cholesky54::shared_memory_size;
    constexpr unsigned int kLambdaOffset   = kBsOffset + sizeof(DType) * kN * kLDA;
    constexpr unsigned int kLambdaBytes    = sizeof(PType) * kN;
    constexpr unsigned int kWorkspaceOffset =
        ((kLambdaOffset + kLambdaBytes) + alignof(DType) - 1) & ~(alignof(DType) - 1);

    DType* As           = reinterpret_cast<DType*>(smem_raw);
    DType* Bs           = reinterpret_cast<DType*>(smem_raw + kBsOffset);
    PType* lambda_s     = reinterpret_cast<PType*>(smem_raw + kLambdaOffset);
    DType* workspace_s  = reinterpret_cast<DType*>(smem_raw + kWorkspaceOffset);

    const DType* L_typed = reinterpret_cast<const DType*>(L_b);
    const DType* M_typed = reinterpret_cast<const DType*>(M_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x) {
        As[idx] = L_typed[idx];
        Bs[idx] = M_typed[idx];
    }
    __syncthreads();

    Heev54().execute(Bs, kLDA, lambda_s, workspace_s,
                     reinterpret_cast<StatusType*>(info_b));
    __syncthreads();                                      // Bs = V (eigvecs of M)
    Trsm54LeftConj().execute(As, kLDA, Bs, kLDA);        // Bs ← L^{-H} V = U
    __syncthreads();

    for (int i = threadIdx.x; i < kN; i += blockDim.x)
        W_b[i] = lambda_s[i];
    DType* U_typed = reinterpret_cast<DType*>(U_b);
    for (int idx = threadIdx.x; idx < kN * kLDA; idx += blockDim.x)
        U_typed[idx] = Bs[idx];
}

extern "C" void geneig_finish_n54_launch(const cuDoubleComplex* d_L,
                                         const cuDoubleComplex* d_M,
                                         double*                d_W,
                                         cuDoubleComplex*       d_U,
                                         int*                   d_info,
                                         int                    batch_size,
                                         cudaStream_t           stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(geneig_finish_n54_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kReduceEigSmem);
        attr_set = true;
    }
    if (batch_size <= 0) return;
    geneig_finish_n54_kernel<<<batch_size, Cholesky54::block_dim,
                               kReduceEigSmem, stream>>>(
        d_L, d_M, d_W, d_U, d_info, batch_size);
}

extern "C" {
unsigned int geneig_finish_n54_block_dim_x()      { return Cholesky54::block_dim.x; }
unsigned int geneig_finish_n54_shared_mem_bytes() { return kReduceEigSmem; }
}
