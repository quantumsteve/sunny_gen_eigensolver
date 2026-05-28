# HybridGeneigSolver tests. All link to the hybrid_geneig static library.

add_executable(test_hybrid_n54 test_hybrid_n54.cpp)
target_link_libraries(test_hybrid_n54
    PRIVATE
        hybrid_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

add_executable(test_hybrid_multiN test_hybrid_multiN.cpp)
target_link_libraries(test_hybrid_multiN
    PRIVATE
        hybrid_geneig
        CUDA::cudart
        CUDA::cuda_driver
)
