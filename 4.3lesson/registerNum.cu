/*********************************************************************************************
 * file name  : resisterNum.cu
 * author     : 权 双
 * date       : 2023-10-29
 * brief      : 测试寄存器使用情况
***********************************************************************************************/

/**
 * GPU硬件资源
 * ------------------------------------------------------------------------------------------------
 * GPU的并行性依靠流多处理器（streaming multiprocessor，或SM），一个GPU是由多个SM构成的。
 * 而SM又由CUDA核心（CUDA core）、共享内存/L1缓存（shared memory/L1 cache）、寄存器文件（RegisterFile）、
 * 加载和存储单元（Load/Store Units）、特殊函数单元（Special Function Unit）、Warps调度（Warps Scheduler）构成。
 * 其中共享内存是整个线程块内的线程之间都可以共享，但无法与其他线程块中的线程共享。特殊函数单元主要用于计算sin, cos, sqrt
 * 等特殊函数。在将一个线程块分配给一个SM时，线程块中的所有线程将会被分成线程束，并由Warps调度器进行调度。
 *
 * 并发与并行的区别：
 * 并发 - 一个核心执行两个任务，并在两个任务中来回切换，但同一个时间点只执行一个任务
 * 并行 - 两个任务同时执行，互不干扰
 *
 * GPU中每个SM中的线程都是并发执行的（而不是并行），因此需要根据GPU的硬件资源来合理分配线程。SM中的Warps调度器
 * 就是用来控制在什么时候执行什么线程的，在同一时间点上，GPU只能执行固定数量的线程，其他未执行的线程则处于等待状态。
 * 在分配任务和计算资源时，系统会以线程块block（而非单个的线程）为单位，向SM分配线程块。当一个线程块被分配到
 * 一个SM上之后，就不能再被分配到其他SM上了。
 *
 * 当线程块被分配到SM上之后，会以32个线程为一组进行分割，每个组被称为一个线程束（warp）。
 * CUDA采用单指令多线程的方式来管理线程。同一个线程块（block）内相邻的32个线程构成一个线程束。
 * 在同一时刻，SM只能执行一个线程束。也就是说，在同一时刻，真正“并行”执行的只有这一个线程束中的32个线程。
 * 注意：每个线程束中只能包含同一个线程块中的线程，即使这个线程块中的线程数量少于32个。因此，为了最大化GPU的执行效率，
 * 我们在定义线程块大小block_size的时候需要使block_size为32的整数倍（32, 64, 128, ...）
 */

/**
 * CUDA内存模型概述（重要）
 * ------------------------------------------------------------------------------------------------
 * 应用程序和GPU在访问数据和代码时，一般不会随机地去访问数据，而是遵循两个局部性原则：
 * 时间局部性 - 如果一个数据在一段时间内被访问，那么它很有可能在短时间内再次被访问，随着时间的推移，被访问的可能性也随之降低
 * 空间局部性 - 如果一段地址被访问，那么这段地址附近的内存可能也会被访问，被访问的概率随着距离的增加而降低
 * 内存层次结构由访问速度从快到慢（但容量从小到大）分别为：寄存器 -> 缓存 -> 主存 -> 磁盘存储器
 * 对于GPU来说，寄存器是稀缺资源，而通常讲的显卡内存一般指主存。需要被经常访问的数据应当储存在寄存器或缓存中，而不经常
 * 访问的数据应当储存在主存或磁盘存储器中。通过编写CUDA程序，我们可以显式地控制这些内存模型的行为。
 *
 * CUDA的内存模型包括：寄存器、共享内存、局部内存、常量内存、纹理内存、全局内存
 * 其中CPU主机内存可以与常量内存，纹理内存，和全局内存进行直接通信（即互相读写）。在GPU中，每个线程块都有一个所有线程可以共享
 * 的共享内存，这意味着每个线程块中的所有线程都可以通过共享内存进行数据交换。共享内存的延迟低，带宽高，因此可以将需要频繁访问
 * 的数据保存在共享内存中。
 * 每个单独的线程都有一个专用的寄存器和局部内存，但局部内存的访问速度相比寄存器要慢很多。
 * 全局内存主要用于GPU设备和CPU主机之间的数据交换，所有线程块中的所有线程都可以和全局内存进行数据交换和读写。全局内存是GPU
 * 中容量最大的内存。
 * 常量内存和纹理内存也都可以和CPU主机之间进行数据交换。GPU中的线程只能读取常量内存和纹理内存中的数据，但不能写入或修改数据。
 * 对于所有类型的GPU，常量内存的容量只有64KB。
 */

/**
 * 寄存器和局部内存
 * ------------------------------------------------------------------------------------------------
 * 寄存器和局部内存的生命周期都是一个线程从开始到结束运行的周期。区别是寄存器在芯片内，而局部内存在芯片外。因此，访问寄存器
 * 数据一般比访问局部内存要快许多。寄存器属于GPU上的稀缺资源，仅在对于单个的线程可见。
 * 对于一个核函数来说，不添加任何限定符（例如__global__, __shared__）的变量和内建变量（例如gridDim, blockDim, blockIdx等）
 * 一般存放于寄存器中。如果变量前添加了__shared__，则该变量会被存放于共享内存中，可以被整个线程块中的所有线程访问。
 * 然而，核函数中不带任何限定符的数组，由于数组可能很大，但寄存器的容量有限，因此有可能存放于寄存器中，也有可能存放于局部内存中。
 * 寄存器都是32位的，因此保存一个double类型的数据需要两个寄存器。寄存器保存在SM的寄存器文件中。
 * 对于计算能力5.0-9.0的GPU，每个SM的寄存器总数量都是64K个，但Fermi架构只有32K个。每个线程块使用的最大数量对于不同架构
 * 是不同的，一般为32KB或64KB。每个线程所独有的寄存器数量最大为255个，但Fermi架构为63个。
 *
 * 如果遇到如下几种情况，数据就会被存放到局部内存中：
 * 1. 数据太大，寄存器放不下
 * 2. 索引值无法在编译时确定的数组
 * 3. 变量不满足寄存器限定条件
 * 对于计算能力5.0-9.0的GPU，每个线程最多可使用512KB的局部内存。本地内存从硬件角度看只是全局内存的一部分，因此访问延迟很高。
 * 过多使用局部内存会降低程序性能。因此，在设计GPU程序时，应尽量使数据存放在寄存器中。但如果无法避免寄存器数据溢出到局部内存，
 * 则可以人为地使局部内存的数据保存到SM的一级缓存或二级缓存中，以提高程序执行的效率。
 *
 * 当核函数所需要的寄存器数量超出硬件设备支持时，数据就会溢出到局部内存中。一般有以下两种情况：
 * 1. 一个SM上并行运行多个线程块/线程束，总共需要的寄存器容量超过64KB
 * 2. 单个线程运行所需要的寄存器数量超过255个
 * 寄存器溢出会降低程序的性能，但若无法避免，则可以将数据保存在GPU的缓存中。
 */

// 可使用nvcc --resource-usage registerNum.cu -o <output_filename> -arch=<arch_name>
#include <stdio.h>
#include "../tools/common.cuh"

__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    // 1、设置GPU设备
    setGPU();

    // 2、分配主机内存和设备内存，并初始化
    int nx = 16;
    int ny = 8;
    int nxy = nx * ny;
    size_t stBytesCount = nxy * sizeof(int);
     
     // （1）分配主机内存，并初始化
    int *ipHost_A, *ipHost_B, *ipHost_C;
    ipHost_A = (int *)malloc(stBytesCount);
    ipHost_B = (int *)malloc(stBytesCount);
    ipHost_C = (int *)malloc(stBytesCount);
    if (ipHost_A != NULL && ipHost_B != NULL && ipHost_C != NULL)
    {
        for (int i = 0; i < nxy; i++)
            {
                ipHost_A[i] = i;
                ipHost_B[i] = i + 1;
            }
        memset(ipHost_C, 0, stBytesCount); 
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }
    

    // （2）分配设备内存，并初始化
    // 这里使用了全局内存，但使用的是动态全局内存，--resource-usage显示的是静态全局内存的使用情况
    int *ipDevice_A, *ipDevice_B, *ipDevice_C;
    ErrorCheck(cudaMalloc((int**)&ipDevice_A, stBytesCount), __FILE__, __LINE__); 
    ErrorCheck(cudaMalloc((int**)&ipDevice_B, stBytesCount), __FILE__, __LINE__); 
    ErrorCheck(cudaMalloc((int**)&ipDevice_C, stBytesCount), __FILE__, __LINE__); 
    if (ipDevice_A != NULL && ipDevice_B != NULL && ipDevice_C != NULL)
    {
        ErrorCheck(cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__); 
        ErrorCheck(cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__); 
        ErrorCheck(cudaMemcpy(ipDevice_C, ipHost_C, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__); 
    }   
    else
    {
        printf("Fail to allocate memory\n");
        free(ipHost_A);
        free(ipHost_B);
        free(ipHost_C);
        exit(1);
    }

    // calculate on GPU
    dim3 block(4, 4);
    dim3 grid((nx + block.x -1) / block.x, (ny + block.y - 1) / block.y);
    printf("Thread config:grid:<%d, %d>, block:<%d, %d>\n", grid.x, grid.y, block.x, block.y);
    
    addMatrix<<<grid, block>>>(ipDevice_A, ipDevice_B, ipDevice_C, nx, ny);  // 调用内核函数
    
    ErrorCheck(cudaMemcpy(ipHost_C, ipDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__); 
    for (int i = 0; i < 10; i++)
    {
        printf("id=%d, matrix_A=%d, matrix_B=%d, result=%d\n", i + 1,ipHost_A[i], ipHost_B[i], ipHost_C[i]);
    }

    free(ipHost_A);
    free(ipHost_B);
    free(ipHost_C);

    ErrorCheck(cudaFree(ipDevice_A), __FILE__, __LINE__); 
    ErrorCheck(cudaFree(ipDevice_B), __FILE__, __LINE__); 
    ErrorCheck(cudaFree(ipDevice_C), __FILE__, __LINE__); 

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__); 
    return 0;
}
