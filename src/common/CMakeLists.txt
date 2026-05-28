# common infrastructure.
#
# Holds the C ABI shim that routes between unified and hybrid paths,
# plus the cuSolverDx header overlay (a path-independent fix for the
# CUSOLVERDX_NVRTC_HEEV_BUG; see ../../docs/CUSOLVERDX_NVRTC_HEEV_BUG.md).
# nvrtc_helper.hpp lives here for any future shared NVRTC utility.

# ---- C ABI shared library --------------------------------------------------
# Routes through either UnifiedGeneigSolver or HybridGeneigSolver based on
# the kernel_mode parameter passed to sunny_geneig_create_ex.
add_library(sunny_geneig SHARED c_api.cpp)

target_include_directories(sunny_geneig
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(sunny_geneig
    PUBLIC
        unified_geneig
        hybrid_geneig
        CUDA::cuda_driver
        CUDA::cudart
)

set_target_properties(sunny_geneig PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
