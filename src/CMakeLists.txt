# layout — src/ aggregates three subdirectory libraries:
#   common/   — NVRTC plumbing helpers, cuSolverDx header overlay, C ABI shim
#   unified/  — UnifiedGeneigSolver: single-launch fused 5-stage NVRTC kernel
#   hybrid/   — HybridGeneigSolver: NVRTC reduce-only + cuSOLVER + cuBLAS
#
# Plus the static-link reference path retained from :
#   geneig_fixed (geneig_fixed.cu) — N=54 hardcoded device kernels
#   smoke (smoke.cu)               — minimal cuSolverDx smoke target

# ---- static-link path ---------------------------------------------
add_executable(smoke smoke.cu)
target_link_libraries(smoke
    PRIVATE
        MathDx::cusolverdx
        CUDA::cudart
)
set_target_properties(smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_library(geneig_fixed STATIC geneig_fixed.cu)
target_link_libraries(geneig_fixed
    PUBLIC
        MathDx::cusolverdx
        CUDA::cudart
)
set_target_properties(geneig_fixed PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ---- Common NVRTC compile-define block ------------------------------------
# Both unified and hybrid translation units need the same set of
# `--include-path` and fatbin-path defines. Encapsulate as a function so
# subdirectory CMakeLists can apply it without duplication.
function(sunny_geneig_apply_nvrtc_defs target visibility)
    list(GET CUDAToolkit_INCLUDE_DIRS 0 FIRST_ELEMENT)
    list(GET CUDAToolkit_INCLUDE_DIRS 1 SECOND_ELEMENT)
    target_compile_definitions(${target}
        ${visibility}
            CUSOLVERDX_FATBIN_PATH="${MathDx_FATBIN}"
            CUSOLVERDX_INCLUDE_DIR="${MathDx_INCLUDE_DIR}"
            CUSOLVERDX_CUTLASS_INCLUDE_DIR="${MathDx_CUTLASS_INCLUDE_DIR}"
            CUDA_INCLUDE_DIR="${FIRST_ELEMENT}"
	        CCCL_INCLUDE_DIR="${SECOND_ELEMENT}"
    )
endfunction()

# ---- 3.5j path libraries --------------------------------------------------
add_subdirectory(unified)
add_subdirectory(hybrid)
add_subdirectory(common)

# ---- C ABI shared library (lives in common/) ------------------------------
# common/CMakeLists.txt builds `sunny_geneig` SHARED that links to both
# unified_geneig and hybrid_geneig.
