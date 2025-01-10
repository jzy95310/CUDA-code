/*********************************************************************************************
 * file name  : query.cu
 * author     : 权 双
 * date       : 2023-08-13
 * brief      : 运行时API查询GPU信息
***********************************************************************************************/

/**
 * 可以使用CUDA运行时API：cudaGetDeviceProperties，来查看每个GPU核心的信息
 */

#include "../tools/common.cuh"
#include <stdio.h>

int main(void)
{
    int device_id = 0;   // GPU设备索引号
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop;   // 需要先初始化一个结构体变量作为参数，传递给cudaGetDeviceProperties运行时API
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);   // 注意这里需要传入结构体prop的地址

    printf("Device id:                                 %d\n",
        device_id);
    printf("Device name:                               %s\n",
        prop.name);
    printf("Compute capability:                        %d.%d\n",
        prop.major, prop.minor);
    printf("Amount of global memory:                   %g GB\n",
        prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Amount of constant memory:                 %g KB\n",   // 常量内存
        prop.totalConstMem  / 1024.0);
    printf("Maximum grid size:                         %d %d %d\n",   // 最大网格大小
        prop.maxGridSize[0], 
        prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                        %d %d %d\n",   // 最大线程块大小
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
    printf("Number of SMs:                             %d\n",   // 多处理器数量
        prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block: %g KB\n",
        prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:    %g KB\n",
        prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Maximum number of registers per block:     %d K\n",   // 每个线程块最大寄存器数量
        prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:        %d K\n",   // 每个多处理器最大寄存器数量
        prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:       %d\n",
        prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:          %d\n",
        prop.maxThreadsPerMultiProcessor);

    return 0;
}