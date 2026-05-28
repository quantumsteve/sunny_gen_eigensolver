// =============================================================================
// LOCAL OVERLAY of cusolverdx/detail/solver_execution.hpp.
//
// This file is a verbatim copy of the upstream header from cuSolverDx 25.12
// with two targeted edits inside `block_execution<Operators...>` (search for
// "[OVERLAY]" below). Both edits address the same NVRTC limitation: NVRTC
// does not implement the C++ class-scope deferred-lookup rule that nvcc
// applies to member-function bodies, so any unqualified identifier inside a
// member function must be lexically declared earlier in the class body.
//
// Bug 1: heev/gesvd `get_workspace_size()` references the unqualified
//        identifier `max_threads_per_block`, but upstream declares it AFTER
//        `workspace_size` (whose initializer calls `get_workspace_size()`).
//        Under NVRTC `max_threads_per_block` is reported as undefined.
//        Fix: declare `max_threads_per_block` BEFORE `workspace_size`.
//
// Bug 2: `get_suggested_batches_per_block()`, `get_suggested_block_dim()`,
//        and `get_workspace_size()` reference the unqualified aliases
//        `job` and `sm` (and others), which upstream declares at the very
//        BOTTOM of the class. Same NVRTC failure mode — substitution
//        failure with `<expression>` markers for `job` and `sm` template
//        arguments. Fix: move the alias block (type/a_arrangement/
//        b_arrangement/transpose/fill_mode/side/diag/job/sm) up to right
//        after `batches_per_block`.
//
// Both fixes are mechanical reorders, no semantic change. nvcc tolerates
// either order; NVRTC requires this order. Verified end-to-end via
// tests/repro_heev_nvrtc.cpp (fails without overlay, succeeds with it).
//
// References:
//   docs/CUSOLVERDX_NVRTC_HEEV_BUG.md  — full diagnosis, repros, suggested
//                                        upstream patch.
//
// Removal:
//   This overlay is a temporary local fix pending an upstream cuSolverDx
//   release that addresses the declaration order. To remove:
//     1. Delete src/common/cusolverdx_overlay/.
//     2. Remove `CUSOLVERDX_OVERLAY_INCLUDE_DIR` define from
//        src/CMakeLists.txt's `target_compile_definitions(nvrtc_geneig ...)`.
//     3. Remove the corresponding `--include-path=` injection (which appears
//        FIRST in the option list, shadowing upstream) in
//        src/unified/unified_solver.cpp.
// =============================================================================

// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUSOLVERDX_DETAIL_EXECUTION_HPP
#define CUSOLVERDX_DETAIL_EXECUTION_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "cusolverdx/detail/solver_description.hpp"
#include "cusolverdx/detail/util.hpp"
#include "cusolverdx/detail/system_checks.hpp"

// BLAS
#include "cusolverdx/database/trsm.cuh"

// Gaussian Elimination
#include "cusolverdx/database/cholesky.cuh"
#include "cusolverdx/database/lu_np.cuh"
#include "cusolverdx/database/lu_pp.cuh"
#include "cusolverdx/database/gtsv_no_pivot.cuh"
#include "cusolverdx/database/trs.cuh"

// QR
#include "cusolverdx/database/qr.cuh"
#include "cusolverdx/database/geqrs.cuh"
#include "cusolverdx/database/unmqr.cuh"
#include "cusolverdx/database/ungqr.cuh"

// Eigenvalue
#include "cusolverdx/database/htev.cuh"
#include "cusolverdx/database/heev.cuh"

// SVD
#include "cusolverdx/database/bdsvd.cuh"
#include "cusolverdx/database/gesvd.cuh"


namespace cusolverdx {
    namespace detail {

        inline static constexpr unsigned smem_align(const unsigned size, const unsigned alignment = 16) {
            return (size + alignment - 1) / alignment * alignment; }

        template<class... Operators>
        class solver_execution: public solver_description<Operators...>, public commondx::detail::execution_description_expression {
            using base_type = solver_description<Operators...>;
            using this_type = solver_execution<Operators...>;

        protected:
            // Precision type
            using typename base_type::this_solver_precision;

            // Value type
            using this_solver_data_type = map_value_type<base_type::this_solver_type_v, this_solver_precision>;

            /// ---- Constraints
            // None

        public:
            using a_data_type = typename this_solver_data_type::a_type;
            using x_data_type = typename this_solver_data_type::x_type;
            using b_data_type = typename this_solver_data_type::b_type;

            static constexpr auto m_size = base_type::this_solver_size::m;
            static constexpr auto n_size = base_type::this_solver_size::n;
            static constexpr auto k_size = base_type::this_solver_size::k;
            static constexpr auto lda    = base_type::this_solver_lda;
            static constexpr auto ldb    = base_type::this_solver_ldb;

            static constexpr auto a_size = base_type::this_solver_a_size;
            static constexpr auto b_size = base_type::this_solver_b_size;

            static constexpr bool is_function_cholesky         = base_type::is_function_cholesky;
            static constexpr bool is_function_lu               = base_type::is_function_lu;
            static constexpr bool is_function_lu_no_pivot      = base_type::is_function_lu_no_pivot;
            static constexpr bool is_function_lu_partial_pivot = base_type::is_function_lu_partial_pivot;
            static constexpr bool is_function_qr               = base_type::is_function_qr;
            static constexpr bool is_function_unmq             = base_type::is_function_unmq;
            static constexpr bool is_function_ungq             = base_type::is_function_ungq;
            static constexpr bool is_function_symmetric_eigen  = base_type::is_function_symmetric_eigen;
            static constexpr bool is_function_solver           = base_type::is_function_solver;
            static constexpr bool is_function_trsm             = base_type::is_function_trsm;
            static constexpr bool is_function_gtsv_no_pivot    = base_type::is_function_gtsv_no_pivot;
        };


        //=============================
        // Block execution
        //=============================
        template<class... Operators>
        class block_execution: public solver_execution<Operators...> {
            using this_type = block_execution<Operators...>;
            using base_type = solver_execution<Operators...>;

            // Import precision type from base class
            using typename base_type::this_solver_precision;

            /// ---- Constraints
            static_assert(base_type::has_block, "Can't create block cusolverdx block execution  without block execution operators");

        public:
            static constexpr auto m_size = base_type::m_size;
            static constexpr auto n_size = base_type::n_size;
            static constexpr auto k_size = base_type::k_size;
            static constexpr auto lda    = base_type::lda;
            static constexpr auto ldb    = base_type::ldb;

            static constexpr unsigned int batches_per_block = base_type::this_solver_batches_per_block_v;

            // [OVERLAY] Aliases for base_type values used by the private
            // get_suggested_*/get_workspace_size member functions below.
            // Upstream declares these at the bottom of the class; under NVRTC
            // the unqualified references in those member-function bodies fail
            // to resolve (no class-scope deferred lookup). Same bug pattern
            // as max_threads_per_block. See file-top comment block.
            static constexpr auto type          = base_type::this_solver_type_v;
            static constexpr auto a_arrangement = base_type::this_solver_arrangement_a;
            static constexpr auto b_arrangement = base_type::this_solver_arrangement_b;
            static constexpr auto transpose     = base_type::this_solver_transpose_v;
            static constexpr auto fill_mode     = base_type::this_solver_fill_mode_v;
            static constexpr auto side          = base_type::this_solver_side_v;
            static constexpr auto diag          = base_type::this_solver_diag_v;
            static constexpr auto job           = base_type::this_solver_job_v;
            static constexpr auto sm            = base_type::this_sm_v;

        private:
            __host__ __device__ __forceinline__ static constexpr unsigned int get_suggested_batches_per_block() {
                static_assert(base_type::is_complete_v, "Can't provide suggested batches per block, description is not complete");
                if constexpr (base_type::is_function_cholesky) {
                    return cholesky::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::this_sm_v>();
                } else if constexpr (base_type::is_function_lu_no_pivot && !base_type::is_function_gtsv_no_pivot) {
                    return lu_np::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::n_size, base_type::this_sm_v>();
                } else if constexpr (base_type::is_function_lu_partial_pivot) {
                    return lu_pp::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::n_size, base_type::this_sm_v>();
                } else if constexpr (base_type::is_function_gtsv_no_pivot) {
                    return gtsv_no_pivot::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::k_size, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::unmqr || function_of_v<this_type> == function::unmlq) {
                    return unmqr::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::n_size, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::ungqr || function_of_v<this_type> == function::unglq) {
                    return ungqr::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::n_size, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::gels) {
                    // Choice of QR or LQ in GELS is based on which dim is larger
                    constexpr unsigned eff_m = const_max(base_type::m_size, base_type::n_size);
                    constexpr unsigned eff_n = const_max(base_type::n_size, base_type::m_size);
                    return geqrs::suggested_batches<a_cuda_data_type, eff_m, eff_n, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::geqrf || function_of_v<this_type> == function::gelqf) { 
                    constexpr unsigned eff_m = const_max(base_type::m_size, base_type::n_size);
                    constexpr unsigned eff_n = const_max(base_type::n_size, base_type::m_size);
                    return qr::suggested_batches<a_cuda_data_type, eff_m, eff_n, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::htev) {
                    return htev::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::this_solver_job_v, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::heev) {
                    return heev::suggested_batches<a_cuda_data_type, base_type::m_size, job, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::bdsvd) {
                    return bdsvd::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::gesvd) {
                    return gesvd::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::n_size, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::trsm) {
                    return trsm::suggested_batches<a_cuda_data_type, base_type::m_size, base_type::n_size, (base_type::this_solver_side_v == side::left), base_type::this_sm_v>();
                } else {
                    return 1;
                }
            }

            __host__ __device__ __forceinline__ static constexpr dim3 get_suggested_block_dim() {
                static_assert(base_type::is_complete_v, "Can't provide suggested block dimensions, description is not complete");
                if constexpr (base_type::is_function_cholesky) {
                    return cholesky::suggested_block_dim<a_cuda_data_type, base_type::m_size, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (base_type::is_function_lu_no_pivot && !base_type::is_function_gtsv_no_pivot) {
                    return lu_np::suggested_block_dim<a_cuda_data_type, base_type::m_size, base_type::n_size, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (base_type::is_function_lu_partial_pivot) {
                    return lu_pp::suggested_block_dim<a_cuda_data_type, base_type::m_size, base_type::n_size, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (base_type::is_function_gtsv_no_pivot) {
                    return gtsv_no_pivot::suggested_block_dim<a_cuda_data_type, base_type::m_size, base_type::k_size, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::unmqr || function_of_v<this_type> == function::unmlq) {
                    return unmqr::suggested_block_dim<a_cuda_data_type, base_type::m_size, base_type::n_size, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::ungqr || function_of_v<this_type> == function::unglq) {
                    return ungqr::suggested_block_dim<a_cuda_data_type, base_type::m_size, base_type::n_size, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (base_type::is_function_qr) { // geqrf, gelqf and gels
                    // Choice of QR or LQ in GELS is based on which dim is larger
                    constexpr unsigned eff_m = const_max(base_type::m_size, base_type::n_size);
                    constexpr unsigned eff_n = const_max(base_type::n_size, base_type::m_size);
                    return qr::suggested_block_dim<a_cuda_data_type, eff_m, eff_n, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::htev) {
                    return htev::suggested_block_dim<a_cuda_data_type, base_type::m_size, base_type::this_solver_job_v, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::heev) {
                    return heev::suggested_block_dim<a_cuda_data_type, base_type::m_size, batches_per_block, job, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::bdsvd) {
                    return bdsvd::suggested_block_dim<a_cuda_data_type, base_type::m_size, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::gesvd) {
                    return gesvd::suggested_block_dim<a_cuda_data_type, base_type::m_size, base_type::n_size, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::trsm) {
                    return trsm::suggested_block_dim<a_cuda_data_type, (base_type::this_solver_side_v == side::left ? base_type::m_size : base_type::n_size), batches_per_block, base_type::this_sm_v>();
                } else {
                    return 256;
                }
            }

            __host__ __device__ __forceinline__ static constexpr dim3 get_block_dim() {
                static_assert(base_type::is_complete_v, "Can't provide block dimensions, description is not complete");
                if constexpr (base_type::has_block_dim) {
                    return base_type::this_block_dim_v;
                }
                return get_suggested_block_dim();
            }

            __host__ __device__ __forceinline__ static constexpr int get_workspace_size() {
                if constexpr (function_of_v<this_type> == function::heev) {
                    return heev::workspace_size<a_cuda_data_type, base_type::m_size, job, max_threads_per_block, batches_per_block, base_type::this_sm_v>();
                } else if constexpr (function_of_v<this_type> == function::gesvd) {
                    return gesvd::workspace_size<a_cuda_data_type, base_type::m_size, base_type::n_size, max_threads_per_block, batches_per_block, base_type::this_sm_v>();
                } else {
                    return 0;
                }
            }

            __device__ __forceinline__ unsigned get_thread_id() {
                const auto dim = get_block_dim();
                __builtin_assume(threadIdx.x < dim.x);
                __builtin_assume(threadIdx.y < dim.y);
                __builtin_assume(threadIdx.z < dim.z);

                return threadIdx.x + dim.x * (threadIdx.y + dim.y * threadIdx.z);
            }

        public:
            inline static constexpr unsigned int get_shared_memory_size() { return get_shared_memory_size(lda, ldb); }

            // support both compile-time and run-time leading dimensions
            inline static constexpr unsigned int get_shared_memory_size(const unsigned int runtime_lda, const unsigned int runtime_ldb = ldb) {
                static_assert(base_type::is_complete_v, "Can't calculate shared memory, description is not complete");

                const unsigned int size_a    = smem_align(sizeof(a_data_type) * runtime_lda * ((base_type::this_solver_arrangement_a == arrangement::col_major) ? n_size : m_size));
                const unsigned int size_b    = smem_align(sizeof(b_data_type) * runtime_ldb * ((base_type::this_solver_arrangement_b == arrangement::col_major) ? k_size : const_max(m_size, n_size)));
                const unsigned int size_ipiv = smem_align(sizeof(int) * const_min(m_size, n_size));
                const unsigned int size_tau  = smem_align(sizeof(a_data_type) * const_min(m_size, n_size));
                const unsigned int size_work = smem_align(sizeof(a_data_type) * get_workspace_size());

                switch (base_type::this_solver_function_v) {
                    case function::potrf:
                    case function::getrf_no_pivot:
                        return batches_per_block * size_a;

                    case function::potrs:
                    case function::posv:
                    case function::getrs_no_pivot:
                    case function::gesv_no_pivot:
                        return batches_per_block * (size_a + size_b);

                    case function::getrf_partial_pivot:
                        return batches_per_block * (size_a + size_ipiv);

                    case function::getrs_partial_pivot:
                    case function::gesv_partial_pivot:
                        return batches_per_block * (size_a + size_b + size_ipiv);

                    case function::gtsv_no_pivot: {
                        const unsigned int size_a_act = 2 * smem_align(sizeof(a_data_type) * (m_size - 1)) + smem_align(sizeof(a_data_type) * m_size);
                        return batches_per_block * (size_a_act + size_b);
                    }

                    case function::geqrf:
                    case function::gelqf:
                        return batches_per_block * (size_a + size_tau);

                    case function::gels:
                        return batches_per_block * (size_a + size_tau + size_b);

                    case function::heev: {
                        const unsigned int size_evals = smem_align(sizeof(a_precision) * m_size);
                        // size_work already includes BPB
                        return batches_per_block * (size_a + size_evals) + size_work;
                    }

                    case function::gesvd: {
                        const unsigned int size_svals = smem_align(sizeof(a_precision) * (m_size < n_size ? m_size : n_size));
                        // size_work already includes BPB
                        return batches_per_block * (size_a + size_svals) + size_work;
                    }

                    // trsm uses m, n, k differently from most routines.
                    // B (m, n). A (m, m) if left and (n, n) if right. k_size is ignored
                    case function::trsm: {
                        const auto AM         = (base_type::this_solver_side_v == side::left) ? m_size : n_size;
                        const auto size_a_act = smem_align(sizeof(a_data_type) * runtime_lda * AM);
                        const auto size_b_act = smem_align(sizeof(b_data_type) * runtime_ldb * ((base_type::this_solver_arrangement_b == arrangement::col_major) ? n_size : m_size));

                        return batches_per_block * (size_a_act + size_b_act);
                    }

                    // unmqr and unmlq use m, n, k differently from most routines.
                    // Matrix C (m, n)
                    // For unmqr: Matrix A/Q (m, k) if left and (n, k) if right
                    // For unmlq: Matrix A/Q (k, m) if left and (k, n) if right
                    case function::unmqr:
                    case function::unmlq:
                        {
                            const bool is_left = (base_type::this_solver_side_v == cusolverdx::side::left);
                            const bool is_qr = base_type::this_solver_function_v == function::unmqr;
                            const auto AM = is_qr ? (is_left ? m_size : n_size) : k_size;
                            const auto AN = is_qr ? k_size : (is_left ? m_size : n_size);
                            const auto BM = m_size;
                            const auto BN = n_size;
                            const unsigned int size_a_act   = smem_align(sizeof(a_data_type) * runtime_lda * ((base_type::this_solver_arrangement_a == arrangement::col_major) ? AN : AM));
                            const unsigned int size_b_act   = smem_align(sizeof(b_data_type) * runtime_ldb * ((base_type::this_solver_arrangement_b == arrangement::col_major) ? BN : BM));
                            const unsigned int size_tau_act = smem_align(sizeof(a_data_type) * k_size);

                            return batches_per_block * (size_a_act + size_tau_act + size_b_act);
                        }
                    
                    case function::ungqr:
                    case function::unglq: {
                        const unsigned int size_tau_act = smem_align(sizeof(a_data_type) * k_size);
                        return batches_per_block * (size_a + size_tau_act);
                    }
                    
                    // htev and bdsvd use m, n, k differently from most routines.
                    // Matrix A is either bidiagonal or symmetric tridiagonal.
                    // D array is of size m, and E array is of size m-1
                    // n and k are ignored
                    case function::htev:
                    case function::bdsvd:
                        {
                            // Need n words for the diagonal and n-1 words for the off-diagonal
                            // If vectors are requested, need n*n additional words
                            int vector_size = (base_type::this_solver_job_v == cusolverdx::job::no_vectors || base_type::this_solver_function_v == function::bdsvd) 
                                              ? 0 : smem_align(sizeof(a_data_type) * m_size * runtime_lda);
                            return batches_per_block * (smem_align(sizeof(a_data_type) * m_size) + smem_align(sizeof(a_data_type) * (m_size - 1)) +
                                                        (base_type::this_solver_job_v == cusolverdx::job::no_vectors ? 0 : smem_align(sizeof(a_data_type) * m_size * runtime_lda)));
                        }

                    default:
                        // Unknown routine
                        return 0;
                }
            }

            // Import value types from base class
            using typename base_type::a_data_type;
            using typename base_type::b_data_type;
            using typename base_type::x_data_type;

            using a_cuda_data_type = typename convert_to_cuda_type<a_data_type>::type;
            using b_cuda_data_type = typename convert_to_cuda_type<b_data_type>::type;
            using x_cuda_data_type = typename convert_to_cuda_type<x_data_type>::type;

            using a_precision = typename base_type::this_solver_precision::a_type;
            using b_precision = typename base_type::this_solver_precision::b_type;
            using x_precision = typename base_type::this_solver_precision::x_type;

            using status_type = int;

            // trsm
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::trsm, void> {

                static_assert(base_type::has_side, "trsm requires a side");
                static_assert(base_type::has_diag, "trsm requires a diagonal mode");
                static_assert(base_type::has_fill_mode, "trsm requires a fill mode");

                trsm::dispatch<a_cuda_data_type, m_size, n_size, base_type::this_solver_side_v, base_type::this_solver_diag_v, base_type::this_solver_transpose_v, base_type::this_solver_fill_mode_v, base_type::this_solver_arrangement_a, base_type::this_solver_arrangement_b, max_threads_per_block, batches_per_block>(
                    (const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // potrf
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrf, void> {
                
                static_assert(base_type::has_fill_mode, "potrf requires a fill mode");

                cholesky::dispatch<a_cuda_data_type, m_size, base_type::this_solver_fill_mode_v, base_type::this_solver_arrangement_a, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (a_cuda_data_type*)A, runtime_lda, status, get_thread_id());
            }

            // potrs
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrs, void> {
                
                static_assert(base_type::has_fill_mode, "potrs requires a fill mode");

                potrs_dispatch<a_cuda_data_type, m_size, k_size, base_type::this_solver_fill_mode_v, base_type::this_solver_arrangement_a, base_type::this_solver_arrangement_b, max_threads_per_block, batches_per_block>(
                    (const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // posv
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb, status_type* status)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv, void> {
                
                static_assert(base_type::has_fill_mode, "posv requires a fill mode");

                cholesky::dispatch<a_cuda_data_type, m_size, base_type::this_solver_fill_mode_v, base_type::this_solver_arrangement_a, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (a_cuda_data_type*)A, runtime_lda, status, get_thread_id());
                potrs_dispatch<a_cuda_data_type, m_size, k_size, base_type::this_solver_fill_mode_v, base_type::this_solver_arrangement_a, base_type::this_solver_arrangement_b, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // getrf_no_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, status_type* status)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_no_pivot, void> {

                lu_np::dispatch<a_cuda_data_type, base_type::m_size, base_type::n_size, base_type::this_solver_arrangement_a, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (a_cuda_data_type*)A, runtime_lda, status, get_thread_id());
            }

            // getrs_no_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb = ldb)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrs_no_pivot, void> {

                static_assert(m_size == n_size, "getrs requires M=N");

                getrs_no_pivot_dispatch<a_cuda_data_type, m_size, k_size, base_type::this_solver_arrangement_a, base_type::this_solver_arrangement_b, base_type::this_solver_transpose_v, max_threads_per_block, batches_per_block>(
                    (const a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // gesv_no_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, b_data_type* B, const unsigned int runtime_ldb, status_type* status)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_no_pivot, void> {

                static_assert(m_size == n_size, "gesv requires M=N");

                // ---- Bug Checks
#if AFFECTED_BY_NVBUG_5288270 && !defined(CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT)
                static constexpr bool can_be_impacted_by_nvbug_5288270 =
                    (base_type::this_sm_v == 1200 && base_type::this_solver_type_v == type::real);

                static_assert(not can_be_impacted_by_nvbug_5288270,
                              "This configuration can be impacted by CUDA 12.8, 12.9 and 13.0 on SM120 for bug 5288270. \n"
                              "Please either update to the latest CUDA version, \n"
                              "or define CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT to ignore this check, \n"
                              "add -Xptxas \"-O0\" to build your application, \n"
                              "and verify correctness of the results every time. \n"
                              "For more details please consult cuSolverDx documentation at https://docs.nvidia.com/cuda/cusolverdx/release_notes.html");
#endif

                lu_np::dispatch<a_cuda_data_type, base_type::m_size, base_type::n_size, base_type::this_solver_arrangement_a, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (a_cuda_data_type*)A, runtime_lda, status, get_thread_id());

                getrs_no_pivot_dispatch<a_cuda_data_type, m_size, k_size, base_type::this_solver_arrangement_a, base_type::this_solver_arrangement_b, base_type::this_solver_transpose_v, max_threads_per_block, batches_per_block>(
                    (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // getrf_partial_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, int* ipiv, status_type* status)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_partial_pivot, void> {

                lu_pp::dispatch<a_cuda_data_type, m_size, n_size, base_type::this_solver_arrangement_a, max_threads_per_block, batches_per_block>((a_cuda_data_type*)A, runtime_lda, ipiv, status, get_thread_id());
            }

            // getrs_partial_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* A, const unsigned int runtime_lda, const int* ipiv, b_data_type* B, const unsigned int runtime_ldb = ldb)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrs_partial_pivot, void> {

                static_assert(m_size == n_size, "getrs requires M=N");

                getrs_partial_pivot_dispatch<a_cuda_data_type, m_size, k_size, base_type::this_solver_arrangement_a, base_type::this_solver_arrangement_b, base_type::this_solver_transpose_v, max_threads_per_block, base_type::this_solver_batches_per_block_v>((const a_cuda_data_type*)A, runtime_lda, ipiv, (b_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // gesv_partial_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, int* ipiv, b_data_type* B, const unsigned int runtime_ldb, status_type* status)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, void> {

                static_assert(m_size == n_size, "gesv requires M=N");

                lu_pp::dispatch<a_cuda_data_type, m_size, n_size, base_type::this_solver_arrangement_a, max_threads_per_block, batches_per_block>((a_cuda_data_type*)A, runtime_lda, ipiv, status, get_thread_id());

                getrs_partial_pivot_dispatch<a_cuda_data_type, m_size, k_size, base_type::this_solver_arrangement_a, base_type::this_solver_arrangement_b, base_type::this_solver_transpose_v, max_threads_per_block, batches_per_block>((a_cuda_data_type*)A, runtime_lda, ipiv, (b_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // gtsv_no_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* dl, const a_data_type* d, const a_data_type* du, a_data_type* b, const unsigned int runtime_ldb, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gtsv_no_pivot, void> {

                gtsv_no_pivot::dispatch<a_cuda_data_type, m_size, k_size, base_type::this_solver_arrangement_b, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (const a_cuda_data_type*)dl, (const a_cuda_data_type*)d, (const a_cuda_data_type*)du, (a_cuda_data_type*)b, runtime_ldb, status, get_thread_id());
            }

            // geqrf
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::geqrf, void> {

                qr::dispatch<a_cuda_data_type, m_size, n_size, base_type::this_solver_arrangement_a, max_threads_per_block, base_type::this_solver_batches_per_block_v>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, get_thread_id());
            }

            // gelqf
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gelqf, void> {

                lq::dispatch<a_cuda_data_type, m_size, n_size, base_type::this_solver_arrangement_a, max_threads_per_block, base_type::this_solver_batches_per_block_v>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, get_thread_id());
            }

            // unmqr
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmqr, void> {

                static_assert(base_type::has_side, "unmqr requires a side");

                unmqr::dispatch<a_cuda_data_type, m_size, n_size, k_size, side, transpose, a_arrangement, b_arrangement, max_threads_per_block, batches_per_block>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }
                    
            // unmlq
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* A, const unsigned int runtime_lda, const a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmlq, void> {

                static_assert(base_type::has_side, "unmlq requires a side");

                unmlq::dispatch<a_cuda_data_type, m_size, n_size, k_size, side, transpose, a_arrangement, b_arrangement, max_threads_per_block, batches_per_block>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, (a_cuda_data_type*)B, runtime_ldb, get_thread_id());
            }

            // ungqr
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::ungqr, void> {
                ungqr::dispatch<a_cuda_data_type, m_size, n_size, k_size, base_type::this_solver_arrangement_a, max_threads_per_block, base_type::this_solver_batches_per_block_v>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, get_thread_id());
            }

            // unglq
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unglq, void> {
                unglq::dispatch<a_cuda_data_type, m_size, n_size, k_size, base_type::this_solver_arrangement_a, max_threads_per_block, base_type::this_solver_batches_per_block_v>((a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, get_thread_id());
            }

            // gels
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gels, void> {

                using T = a_cuda_data_type;
                constexpr auto arr_a = base_type::this_solver_arrangement_a;
                constexpr auto arr_b = base_type::this_solver_arrangement_b;
                constexpr auto bpb   = base_type::this_solver_batches_per_block_v;

                auto tid = get_thread_id();

                // Use QR for tall-skinny and LQ for short-wide
                if constexpr (m_size >= n_size) {
                    qr::dispatch<T, m_size, n_size, arr_a, max_threads_per_block, bpb>(
                        (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, tid);

                    geqrs::dispatch<T, m_size, n_size, k_size, arr_a, arr_b, base_type::this_solver_transpose_v, max_threads_per_block, bpb>(
                        (const a_cuda_data_type*)A, runtime_lda, (const a_cuda_data_type*)tau, (b_cuda_data_type*)B, runtime_ldb, tid);
                } else {
                    lq::dispatch<T, m_size, n_size, arr_a, max_threads_per_block, bpb>(
                        (a_cuda_data_type*)A, runtime_lda, (a_cuda_data_type*)tau, tid);

                    gelqs::dispatch<T, m_size, n_size, k_size, arr_a, arr_b, base_type::this_solver_transpose_v, max_threads_per_block, bpb>(
                        (const a_cuda_data_type*)A, runtime_lda, (const a_cuda_data_type*)tau, (b_cuda_data_type*)B, runtime_ldb, tid);
                }
            }

            // htev - optional vectors
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* d, a_data_type* e, a_data_type* v, int ldv, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::htev, void> {
                htev::dispatch<a_cuda_data_type, a_cuda_data_type, m_size, job, base_type::this_solver_arrangement_a, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (a_cuda_data_type*)d, (a_cuda_data_type*)e, (a_cuda_data_type*)v, ldv, status, get_thread_id());
            }

            // htev - no vectors
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* d, a_data_type* e, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::htev, void> {
                static_assert(base_type::this_solver_job_v == job::no_vectors, "To compute vectors with htev, a vector array must be provided.");
                execute(d, e, nullptr, m_size, status);
            }

            // heev
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, a_precision* lambda, a_data_type* workspace, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::heev, void> {

                static_assert(base_type::has_fill_mode, "heev requires a fill mode");
                static_assert(base_type::this_solver_job_v == job::no_vectors || base_type::this_solver_job_v == job::overwrite_vectors);

                heev::dispatch<a_cuda_data_type, m_size, base_type::this_solver_fill_mode_v, base_type::this_solver_arrangement_a, job, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (a_cuda_data_type*)A, lda, lambda, (a_cuda_data_type*)workspace, status, get_thread_id());
            }

            // bdsvd - no vectors
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* d, a_data_type* e, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::bdsvd, void> {
                bdsvd::dispatch<a_cuda_data_type, m_size, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (a_cuda_data_type*)d, (a_cuda_data_type*)e, status, get_thread_id());
            }

            // gesvd - no vectors
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, a_precision* sigma, a_data_type* workspace, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesvd, void> {

                static_assert(base_type::this_solver_job_v == job::no_vectors);

                gesvd::dispatch<a_cuda_data_type, m_size, n_size, base_type::this_solver_arrangement_a, max_threads_per_block, batches_per_block, base_type::this_sm_v>(
                    (a_cuda_data_type*)A, lda, sigma, (a_cuda_data_type*)workspace, status, get_thread_id());
            }


            //======================= Helpers for adding lda/ldb from operator
            // potrf, getrf_no_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::potrf || function_of_v<Solver> == function::getrf_no_pivot, void> {
                execute(A, lda, status);
            }
            // getrf_partial_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, int* ipiv, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrf_partial_pivot, void> {
                execute(A, lda, ipiv, status);
            }

            // trsm, potrs, getrs_no_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* A, b_data_type* B, unsigned int runtime_ldb=ldb) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::trsm || function_of_v<Solver> == function::potrs || function_of_v<Solver> == function::getrs_no_pivot, void> {
                execute(A, lda, B, runtime_ldb);
            }
            // getrs_partial_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const int* ipiv, b_data_type* B, unsigned int runtime_ldb=ldb) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::getrs_partial_pivot, void> {
                execute(A, lda, ipiv, B, runtime_ldb);
            }

            // posv, gesv_no_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, b_data_type* B, status_type* status)  -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv || function_of_v<Solver> == function::gesv_no_pivot, void> {
                execute(A, lda, B, ldb, status);
            }
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, b_data_type* B, const unsigned int runtime_ldb, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv || function_of_v<Solver> == function::gesv_no_pivot, void> {
                execute(A, lda, B, runtime_ldb, status);
            }
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, b_data_type* B, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::posv || function_of_v<Solver> == function::gesv_no_pivot, void> {
                execute(A, runtime_lda, B, ldb, status);
            }
            // gesv_partial_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, int* ipiv, b_data_type* B, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, void> {
                execute(A, lda, ipiv, B, ldb, status);
            }
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, int* ipiv, b_data_type* B, unsigned int runtime_ldb, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, void> {
                execute(A, lda, ipiv, B, runtime_ldb, status);
            }
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, int* ipiv, b_data_type* B, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesv_partial_pivot, void> {
                execute(A, runtime_lda, ipiv, B, ldb, status);
            }

            // gtsv_no_pivot
            template<class Solver = this_type>
            inline __device__ auto execute(const a_data_type* dl, const a_data_type* d, const a_data_type* du, a_data_type* b, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gtsv_no_pivot, void> {
                execute(dl, d, du, b, ldb, status);
            }

            // geqrf, gelqf, ungqr, unglq
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, a_data_type* tau) -> COMMONDX_STL_NAMESPACE::enable_if_t<(function_of_v<Solver> == function::geqrf || function_of_v<Solver> == function::gelqf || function_of_v<Solver> == function::ungqr || function_of_v<Solver> == function::unglq), void> {
                execute(A, lda, tau);
            }

            // unmqr, unmlq, gels
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, a_data_type* tau, b_data_type* B, const unsigned int runtime_ldb = ldb) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmqr || function_of_v<Solver> == function::unmlq || function_of_v<Solver> == function::gels, void> {
                execute(A, lda, tau, B, runtime_ldb);
            }
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, const unsigned int runtime_lda, a_data_type* tau, b_data_type* B) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::unmqr || function_of_v<Solver> == function::unmlq || function_of_v<Solver> == function::gels, void> {
                execute(A, runtime_lda, tau, B, ldb);
            }

            // htev
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* d, a_data_type* e, a_data_type* v, status_type* status) -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::htev, void> {
                execute(d, e, v, lda, status);
            }

            // heev
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, a_precision* lambda, a_data_type* workspace, status_type* status)  -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::heev, void> {
                execute(A, lda, lambda, workspace, status);
            }

            // gesvd
            template<class Solver = this_type>
            inline __device__ auto execute(a_data_type* A, a_precision* sigma, a_data_type* workspace, status_type* status)  -> COMMONDX_STL_NAMESPACE::enable_if_t<function_of_v<Solver> == function::gesvd, void> {
                execute(A, lda, sigma, workspace, status);
            }

            static constexpr unsigned int suggested_batches_per_block = get_suggested_batches_per_block();

            static constexpr dim3 suggested_block_dim = get_suggested_block_dim();
            static constexpr dim3 block_dim           = get_block_dim();

            // [OVERLAY] max_threads_per_block moved up to before workspace_size
            // so heev/gesvd's get_workspace_size() can resolve it under NVRTC.
            // See file-top comment block.
            static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;

            static constexpr unsigned int workspace_size = get_workspace_size();
            static constexpr unsigned int shared_memory_size = get_shared_memory_size();

            static constexpr unsigned int min_blocks_per_multiprocessor = 1;

            // [OVERLAY] type/a_arrangement/b_arrangement/transpose/fill_mode/
            // side/diag/job/sm aliases moved up near batches_per_block so
            // member-function bodies can resolve them under NVRTC.
        };
    } // namespace detail
} // namespace cusolverdx

#endif // CUSOLVERDX_DETAIL_EXECUTION_HPP
