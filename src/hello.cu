// День 2: перший запуск. Кожен потік GPU друкує своє місце в сітці.
#include <cstdio>
#include "cuda_utils.cuh"

__global__ void hello_kernel() {
    printf("hello з блоку %d, потік %d (глобальний індекс %d)\n",
           blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, SM: %d, пам'ять: %.1f ГБ\n", prop.name,
                prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

    hello_kernel<<<2, 4>>>();  // 2 блоки по 4 потоки
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
