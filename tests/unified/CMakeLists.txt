# UnifiedGeneigSolver tests. All link to the unified_geneig static library.

# regression — single-N=54 unified-kernel correctness test.
add_executable(test_n54 test_n54.cpp)
target_link_libraries(test_n54
    PRIVATE
        unified_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# multi-N validation of the unified NVRTC kernel.
add_executable(test_multiN test_multiN.cpp)
target_link_libraries(test_multiN
    PRIVATE
        unified_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# BatchesPerBlock sweep at N=54.
add_executable(test_bpb_sweep_n54 test_bpb_sweep_n54.cpp)
target_link_libraries(test_bpb_sweep_n54
    PRIVATE
        unified_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# BlockDim micro-bench at N=32, BPB=2.
add_executable(test_blockdim_n32 test_blockdim_n32.cpp)
target_link_libraries(test_blockdim_n32
    PRIVATE
        unified_geneig
        LAPACK::LAPACK
        CUDA::cudart
        CUDA::cuda_driver
)

# multi-stream chunked launch benchmark at N=54.
add_executable(test_streams_n54 test_streams_n54.cpp)
target_link_libraries(test_streams_n54
    PRIVATE
        unified_geneig
        LAPACK::LAPACK
        CUDA::cudart
        CUDA::cuda_driver
)

# autotune benchmarks at N=54 BPB=1 and N=32 BPB=2.
add_executable(test_autotune_n54 test_autotune_n54.cpp)
target_link_libraries(test_autotune_n54
    PRIVATE
        unified_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

add_executable(test_autotune_n32 test_autotune_n32.cpp)
target_link_libraries(test_autotune_n32
    PRIVATE
        unified_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

# module cache benchmark and corruption recovery test.
add_executable(test_cache_n54 test_cache_n54.cpp)
target_link_libraries(test_cache_n54
    PRIVATE
        unified_geneig
        CUDA::cudart
        CUDA::cuda_driver
)

add_executable(test_cache_invalidation test_cache_invalidation.cpp)
target_link_libraries(test_cache_invalidation
    PRIVATE
        unified_geneig
        CUDA::cudart
        CUDA::cuda_driver
)
