# Path-agnostic infrastructure tests.
#
# These exercise things that are not specific to either unified or hybrid:
# the C ABI shim, cuSolverDx NVRTC repros, the heev-fraction per-stage
# diagnostic, the standalone cuSOLVER XsyevBatched benchmark, and the
# fixture-generation utilities (lapack_reference, dump_pair).

# Fixture-generation tools.
add_executable(lapack_reference lapack_reference.cpp)
target_link_libraries(lapack_reference PRIVATE LAPACK::LAPACK)

add_executable(dump_pair dump_pair.cpp)
target_link_libraries(dump_pair PRIVATE LAPACK::LAPACK)

# minimal cuSolverDx NVRTC bug repros (heev fails without
# the overlay; potrf succeeds on plain cuSolverDx 25.12).
add_executable(repro_heev_nvrtc  repro_heev_nvrtc.cpp)
add_executable(repro_potrf_nvrtc repro_potrf_nvrtc.cpp)
foreach(_t repro_heev_nvrtc repro_potrf_nvrtc)
    target_link_libraries(${_t} PRIVATE CUDA::nvrtc CUDA::cuda_driver)
    list(GET CUDAToolkit_INCLUDE_DIRS 0 FIRST_ELEMENT)
    list(GET CUDAToolkit_INCLUDE_DIRS 1 SECOND_ELEMENT)
    target_compile_definitions(${_t}
        PRIVATE
            CUSOLVERDX_INCLUDE_DIR="${MathDx_INCLUDE_DIR}"
            CUSOLVERDX_CUTLASS_INCLUDE_DIR="${MathDx_CUTLASS_INCLUDE_DIR}"
            CUDA_INCLUDE_DIR="${FIRST_ELEMENT}"
            CCCL_INCLUDE_DIR="${SECOND_ELEMENT}"
    )
endforeach()

# exercise the C ABI shim libsunny_geneig.so before Julia.
add_executable(test_c_api test_c_api.cpp)
target_link_libraries(test_c_api
    PRIVATE
        sunny_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# standalone cuSOLVER XsyevBatched bench.
add_executable(bench_xsyev_only bench_xsyev_only.cpp)
target_link_libraries(bench_xsyev_only
    PRIVATE
        CUDA::cudart
        CUDA::cusolver
        LAPACK::LAPACK
)

# Steven-comparison — standalone cuSOLVER ZheevjBatched (Jacobi) bench.
add_executable(bench_zheevj_only bench_zheevj_only.cpp)
target_link_libraries(bench_zheevj_only
    PRIVATE
        CUDA::cudart
        CUDA::cusolver
        LAPACK::LAPACK
)

# Steven-comparison — standalone cuSolverDx::Heev-only bench (NVRTC kernel).
add_executable(bench_heev_dx_only bench_heev_dx_only.cpp)
target_link_libraries(bench_heev_dx_only
    PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        CUDA::nvJitLink
        LAPACK::LAPACK
)
sunny_geneig_apply_nvrtc_defs(bench_heev_dx_only PRIVATE)

# One-shot workspace-scaling diagnostic.
add_executable(probe_ws_scaling probe_ws_scaling.cpp)
target_link_libraries(probe_ws_scaling
    PRIVATE
        CUDA::cudart
        CUDA::cusolver
)

# per-stage timing diagnostic
# (full / reduce-only / no-vec). Compiles three kernel variants directly.
add_executable(test_heev_fraction test_heev_fraction.cpp)
target_link_libraries(test_heev_fraction
    PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        CUDA::nvJitLink
)
sunny_geneig_apply_nvrtc_defs(test_heev_fraction PRIVATE)

# probe-compile timing measurement (one-off).
add_executable(probe_compile_timing probe_compile_timing.cpp)
target_link_libraries(probe_compile_timing
    PRIVATE
        CUDA::nvrtc
        CUDA::cuda_driver
        CUDA::nvJitLink
)
sunny_geneig_apply_nvrtc_defs(probe_compile_timing PRIVATE)
