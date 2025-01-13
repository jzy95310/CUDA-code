/**************************************************************************************
 *  file name   : dynamic_shared_memory.cu
 *  author      : 权 双
 *  data        : 2023-12-26
 *  brief       : 动态共享内存使用
***************************************************************************************/

/**
 * 共享内存为片上（on-chip）内存，拥有仅次于寄存器的读取速度，相比于局部内存和全局内存有更高的带宽和更低的延迟。
 * 然而，共享内存的容量也有限，以KB为单位（例如：计算能力8.9的4070显卡共享内存为100KB），因此也是极为稀缺的资源。
 * 共享内存在整个线程块内都可见，因此可以用于线程之间的通信，其生命周期与所属线程块一致。使用__shared__修饰符的
 * 变量存放于共享内存中。共享内存可定义为动态和静态两种。
 * 每个SM中共享内存的数量是一定的。要访问共享内存，必须加入同步机制，即必须以下命令使线程块内的线程同步：
 * void __syncthreads();
 *
 * 共享内存一般有两个作用：
 * 1. 减少核函数对全局内存的访问次数，实现高效通信。经常访问的数据可以从全局内存搬移到共享内存中。
 * 2. 改变全局内存访问内存的事务方式，提高数据访问的带宽。
 *
 * 静态共享内存有两种声明方式：
 * 1. 在核函数中声明，其作用域仅局限在当前核函数中。
 * 2. 在核函数外声明，其作用域对所有核函数有效。
 * 注意：静态共享内存在编译时就要确定内存大小，例如：__shared__ float tile[size, size]，在一般情况下，我们
 * 需要使size等于线程块大小（32或32的倍数）。
 *
 * 相反，在使用动态共享内存时不需要事先确定内存大小。在定义动态共享内存时，必须加上extern关键字
 * 同时，在定义核函数时，需要使用一个额外的参数来定义动态共享内存的大小，即：
 * kernel_fn<<<grid_size,block_size,shared_mem_size>>>(kernel_fn_arguments)
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

extern __shared__ float s_array[];   // 在核函数外定义一个数组在动态共享内存中
// extern __shared__ float *s_array;   不能使用指针来定义动态共享内存

__global__ void kernel_1(float* d_A, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;

    if (n < N)
    {
        s_array[tid] = d_A[n];
    }
    __syncthreads();

    if (tid == 0)
    {
        for (int i = 0; i < 32; ++i)
        {
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
    kernel_1<<<grid, block, 32>>>(d_A, nElems);   // 这里使用了一个额外的参数来定义动态共享内存大小为32

    CUDA_CHECK(cudaFree(d_A));
    free(h_A);
    CUDA_CHECK(cudaDeviceReset());

}