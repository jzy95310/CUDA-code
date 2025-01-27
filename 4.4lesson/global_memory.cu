/*********************************************************************************************
 * file name  :global_memory.cu
 * author     : 权 双
 * date       : 2023-12-26
 * brief      : 静态全局变量使用
***********************************************************************************************/

/**
 * 全局内存是GPU中延迟最大，容量最大，同时使用也最频繁的内存。全局内存中的数据对于所有的线程都可见，对于主机CPU也可见。
 * 全局内存的生命周期由主机CPU端决定，从主机端使用cudaMalloc分配内存时开始，到主机端使用cudaFree释放内存结束。全局
 * 内存保存在DRAM上。
 *
 * 全局内存的初始化分两种：
 * 1. 动态初始化 - 使用CUDA运行时API cudaMalloc动态声明内存空间，由cudaFree释放内存
 * 2. 静态初始化 - 使用__device__修饰符静态声明全局内存，在编译器编译期间就已经确定
 * 需要注意的是，静态全局变量（即带有__device__）需要在所有主机（__host__）和核函数（__global__）外部进行定义。
 * 在核函数中，可以直接对静态全局变量进行访问，主机CPU则不能“直接”访问静态全局变量。然而，主机CPU可以通过下面两个
 * 函数与静态全局内存进行通信：
 * 1. cudaMemcpyToSymbol - 将主机数据（符号变量）传递给静态全局内存
 * 2. cudaMemcpyFromSymbol - 将静态全局内存中的数据（符号变量）传递给主机
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

// 注意：静态全局变量必须在外部定义
__device__ int d_x = 1;   // 定义静态全局变量
__device__ int d_y[2];   // 定义静态全局数组，含有两个元素

__global__ void kernel(void)
{
    d_y[0] += d_x;   // 核函数中可以直接访问静态全局变量
    d_y[1] += d_x;

    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}



int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    int h_y[2] = {10, 20};
    CUDA_CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));   // 将主机中的数据传递给静态全局变量

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));   // 将经过计算后的静态全局变量传递回主机
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);   // 结果为[11, 21]

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

// 注：该程序使用CMake进行编译：
// cmake -S . -B build
// cmake --build build/