// First kernel launch. Every GPU thread prints its place in the grid.
#include <cstdio>
#include "cuda_utils.cuh"

__global__ void hello_kernel() {
    printf("hello from block %d, thread %d (global index %d)\n", blockIdx.x, threadIdx.x,
           blockIdx.x * blockDim.x + threadIdx.x);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, SM: %d, memory: %.1f GB\n", prop.name, prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

    hello_kernel<<<2, 4>>>();  // 2 blocks of 4 threads
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
