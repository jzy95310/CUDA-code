/*********************************************************************************************
 * file name  : global_memory.cu
 * author     : 权 双
 * date       : 2023-12-26
 * brief      : 常量内存的使用
***********************************************************************************************/

/**
 * 常量内存是有常量缓存的全局内存，大小仅为64KB。由于有缓存，线程在读取常量内存中的数据时，速度一般比读取全局内存中的数据快。
 * 常量内存中的数据对所有线程块中的线程都可见。
 * 在定义常量内存中的变量时，需要使用__constant__修饰符。需要注意的是，常量内存必须定义在核函数和主机函数外部，且常量内存
 * 必须采用静态定义（即定义时必须声明大小）。同时，常量内存仅可读，不可写（仅针对GPU中的线程，CPU对于常量内存是可读也可写的）。
 * 我们在给核函数传递参数时，这些参数变量就存放于常量内存中。
 *
 * 与4.4中的静态全局内存一样，在初始化常量内存时，可以在主机端使用cudaMemcpyToSymbol将主机中的数据（符号变量）传递给常量
 * 内存。如果所有线程都需要从同一个内存地址中访问数据（例如数学公式中的系数），那么常量内存表现最好。因为线程只需要读取一次，
 * 然后就可以广播给当前线程束中的所有线程。
 */

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

// 必须在核函数和主机函数之外定义常量内存
__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

__global__ void kernel_1(void)
{
    
    printf("Constant data c_data = %.2f.\n", c_data);
}

__global__ void kernel_2(int N)
{
    int idx = threadIdx.x;
    if (idx < N)   // 这里的数组长度N是传递给核函数的参数，因此存放于常量内存中，因为所有的线程都需要知道数组长度才能与index进行比较
    {

    }   
}

int main(int argc, char **argv)
{ 
    
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    float h_data = 8.8f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));   // 将h_data传递给常量内存中的变量c_data

    dim3 block(1);
    dim3 grid(1);
    kernel_1<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));   // 将常量内存中的变量c_data2复制给主机
    printf("Constant data h_data = %.2f.\n", h_data);

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}