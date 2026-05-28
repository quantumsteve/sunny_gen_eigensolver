# hybrid path (the 3.5i kernel).
#
# HybridGeneigSolver compiles only the reduce stage (potrf + trsm × 2)
# under NVRTC, then dispatches cusolverDnXsyevBatched and
# cublasZtrsmBatched_64 for the eigensolve and back-transform.
# Kernel source in reduce_kernel_source.hpp. See ../../docs/HYBRID_PATH.md.

add_library(hybrid_geneig STATIC hybrid_solver.cpp)

target_include_directories(hybrid_geneig
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(hybrid_geneig
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        CUDA::nvJitLink
        CUDA::cusolver
        CUDA::cublas
)

sunny_geneig_apply_nvrtc_defs(hybrid_geneig PRIVATE)

set_target_properties(hybrid_geneig PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
