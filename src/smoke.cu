#include <cusolverdx.hpp>
#include <cstdio>

using namespace cusolverdx;

// Minimal POSV solver definition to force template instantiation
// Size<M, N, K>: M=N=32 for square A, K=1 right-hand side
using Solver = decltype(Size<32, 32, 1>()
                      + Precision<double>()
                      + Type<type::real>()
                      + Function<posv>()
                      + FillMode<fill_mode::lower>()
                      + Arrangement<arrangement::col_major, arrangement::col_major>()
                      + Block()
                      + BlockDim<32>()
                      + SM<800>());

__global__ void smoke_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("cuSolverDx smoke OK, shared mem bytes: %u\n",
               (unsigned)Solver::shared_memory_size);
    }
}

int main() {
    smoke_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
