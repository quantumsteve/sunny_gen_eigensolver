# unified single-launch path (the 3a/3b/3.5a-f kernel).
#
# UnifiedGeneigSolver compiles a five-stage fused NVRTC kernel
# (potrf + trsm × 2 + heev + back-trsm) per (N, BPB, arch, BlockDim.x);
# kernel source in kernel_source.hpp. See ../../docs/UNIFIED_PATH.md.

add_library(unified_geneig STATIC unified_solver.cpp)

target_include_directories(unified_geneig
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(unified_geneig
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        CUDA::nvJitLink
)

sunny_geneig_apply_nvrtc_defs(unified_geneig PRIVATE)

set_target_properties(unified_geneig PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
