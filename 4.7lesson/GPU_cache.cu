/*********************************************************************************************
 * file name  : GPU_cache.cu
 * author     : 权 双
 * date       : 2023-12-30
 * brief      : GPU缓存的使用
***********************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"


/**
 *
 * GPU缓存的种类
 * --------------------------------------------------------------------------------------------
 * GPU中有4种缓存：
 * 1. 一级缓存（L1） - 每个SM上都有且只有一个L1缓存
 * 2. 二级缓存（L2） - 所有SM共享一个L2缓存
 * 3. 只读常量缓存 - 每个SM都有且只有一个只读常量缓存，用于提高来自常量内存的数据的读取性能
 * 4. 只读纹理缓存 - 每个SM都有且只有一个只读纹理缓存，用于提高来自纹理内存的数据的读取性能
 * GPU缓存是不可编程的内存。当GPU核心从全局内存中读取数据后，会先从DRAM上加载数据，之后经过L2缓存，之后“可能”经过
 * L1缓存（可通过编译选项设置经过或不经过），最后才到达SM执行运算。同时，L1和L2缓存主要用来存储局部内存和全局内存
 * 中的数据，以及寄存器溢出的部分。
 * 特别注意：在GPU中，只有从内存中读取的数据可以被缓存，要写入到内存中的数据是不能被缓存的。
 *
 * L1缓存的查询与设置
 * --------------------------------------------------------------------------------------------
 * 并非所有GPU都支持L1缓存查询，可以使用bool globalL1CacheSupported函数来确定是否支持。在默认设置下，从DRAM
 * 中读取的数据只会经过L2缓存，并不会经过L1缓存，但可以通过以下两条编译指令来启用L1缓存：
 * -Xptxas -dlcm=ca - 除一些特殊的数据（带有禁用缓存修饰符的数据）以外，其他数据都会经过L1缓存
 * -Xptxas -fscm=ca - 所有数据都会经过L1缓存
 * L1缓存一般与共享内存和纹理内存共用一个统一的数据缓存。例如，对于计算能力为8.9的显卡来说，统一数据缓存大小为128KB。
 * 共享内存从统一数据缓存中分区出来，可以通过CUDA函数cudaFuncSetAttribute手动设置共享内存的大小，剩下的空间用作
 * L1缓存或纹理内存。然而，手动设置并不一定会生效，GPU会自动选择最优的配置。
 */


__global__ void kernel(void)
{
    
}


int main(int argc, char **argv)
{ 
    
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    if (deviceProps.globalL1CacheSupported){   // 确认GPU显卡型号是否支持启用L1缓存
        std::cout << "支持全局内存L1缓存" << std::endl;
    }
    else{
        std::cout << "不支持全局内存L1缓存" << std::endl;
    }
    std::cout << "L2缓存大小：" << deviceProps.l2CacheSize / (1024 * 1024) << "M" << std::endl;   // deviceProps.l2CacheSize返回字节数大小

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}