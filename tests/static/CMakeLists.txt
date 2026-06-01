# reference tests — N=54 hardcoded device kernels in geneig_fixed.
#
# These are the canonical correctness anchors against the LAPACK fixtures.
# They predate NVRTC work and exercise the static-link cuSolverDx
# composition pattern.

add_executable(test_chol_n54 test_chol_n54.cpp)
target_link_libraries(test_chol_n54
    PRIVATE
        geneig_fixed
	LAPACK::LAPACK
        CUDA::cudart
)
set_target_properties(test_chol_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_executable(test_reduce_n54 test_reduce_n54.cpp)
target_link_libraries(test_reduce_n54
    PRIVATE
        geneig_fixed
        LAPACK::LAPACK
        CUDA::cudart
)
set_target_properties(test_reduce_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_executable(test_reduce_eig_n54 test_reduce_eig_n54.cpp)
target_link_libraries(test_reduce_eig_n54
    PRIVATE
        geneig_fixed
        LAPACK::LAPACK
        CUDA::cudart
)
set_target_properties(test_reduce_eig_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_executable(test_full_n54 test_full_n54.cpp)
target_link_libraries(test_full_n54
    PRIVATE
        geneig_fixed
        CUDA::cudart
)
set_target_properties(test_full_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)

add_executable(test_full_batched_n54 test_full_batched_n54.cpp)
target_link_libraries(test_full_batched_n54
    PRIVATE
        geneig_fixed
        LAPACK::LAPACK
        CUDA::cudart
)
set_target_properties(test_full_batched_n54 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON
)
