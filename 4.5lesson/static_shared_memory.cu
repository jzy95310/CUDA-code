/**************************************************************************************
 *  file name   : static_shared_memory.cu
 *  author      : 权 双
 *  data        : 2023-12-26
 *  brief       : 静态共享内存使用
***************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

__global__ void kernel_1(float* d_A, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    // 在静态共享内存中定义一个长度为32的数组，这意味着每一个线程块（block）中都会有一个s_array数组。
    // 虽然所有线程块中都有一个名为s_array的数组，但每个线程块都可以对各自的s_array数组进行独立的操作。
    __shared__ float s_array[32];

    if (n < N)
    {
        // 将全局内存中的数据复制给两个线程块中的共享内存
        // 注意这里s_array的索引为threadIdx（tid），即一个线程块中的线程索引。因为共享内存仅对同一个线程块中的所有线程可见。
        // Recall：这里同一个线程块中的所有线程为并行运行，每个线程块负责一个数组元素，因此不需要for循环
        s_array[tid] = d_A[n];
    }
    __syncthreads();   // 同步一个线程块中的所有线程

    if (tid == 0)   // 仅使用每个线程块中的第一个线程来进行打印
    {
        for (int i = 0; i < 32; ++i)
        {
            // 这里仅使用第一个线程就可以打印共享内存中的所有数据（再次强调：共享内存中的数据对同一线程块中的所有线程都可见）
            printf("kernel_1: %f, blockIdx: %d\n", s_array[i], bid);
        }
    }

}



int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    int nElems = 64;
    int nbytes = nElems * sizeof(float);

    float* h_A = nullptr;
    h_A = (float*)malloc(nbytes);
    for (int i = 0; i < nElems; ++i)
    {
        h_A[i] = float(i);
    }

    float* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, nbytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nbytes,cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(2);
    kernel_1<<<grid, block>>>(d_A, nElems);

    CUDA_CHECK(cudaFree(d_A));
    free(h_A);
    CUDA_CHECK(cudaDeviceReset());

}