/**
 * 线程束分化
 * ---------------------------------------------------------------------------------------------
 * 如果同一个线程束中的指令执行了不同分支的指令（例如if/else指令），就会导致线程束分化，从而影响程序执行的并行性。
 * 想要避免线程束分化，在同一个时钟周期内，一个线程束中的所有线程必须执行相同的指令。
 * 然而，不同的线程束中的线程是不会发生线程束分化的。因此，为了解决线程束分化问题，我们可以让不同的线程束执行不同的
 * 指令，例如：if ((tid / 32) % 2 == 0) {...}
 *
 * 并行规约计算
 * ---------------------------------------------------------------------------------------------
 * 在向量中满足交换律与结合律的运算，称为规约问题。并行执行的规约计算称为并行规约计算。例如，要并行计算一个数组所有
 * 元素之和，我们可以使用邻域并行计算和间域并行计算。
 * 1. 邻域并行计算 - 元素与它们直接相邻的元素进行配对
 * 2. 间域并行计算 - 元素与和它们有一定间隔的另一元素进行配对
 * 在进行邻域并行计算时，如果程序设计不当，就会出现严重的线程束分化。例如，假设一个数组有512个元素，我们用一个大小
 * 为512的线程块（即16个线程束）进行求和。如果我们使用全部的线程进行求和，那么索引为偶数和索引为奇数的线程所执行
 * 的指令是不一样的，这样就会导致线程束分化。然而，如果我们一上来只使用前8个线程束（256个线程）进行求和，每一个线程
 * 负责相邻的两个元素，那么所有线程都会执行相同的指令，这样就避免了线程束分化。
 */

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <string>
#include "utils.hpp"
#include "timer.hpp"
#include "reduce.hpp"
#include <cstring>
#include <memory>
#include <cmath>

int seed;
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "用法: ./build/reduction [size] [blockSize]" << std::endl;
        return -1;
    }

    Timer timer;   // 见timer.cpp，定义了一个计算时间的类
    char str[100];
    int size = std::stoi(argv[1]);
    int blockSize = std::stoi(argv[2]);
   
    int gridsize = size / blockSize;   // 一般情况下，这里需要向上取整
    
    float* h_idata = nullptr;
    float* h_odata = nullptr;
    h_idata = (float*)malloc(size * sizeof(float));   // 存储需要计算的数组内容
    h_odata = (float*)malloc(gridsize * sizeof(float));   // 存储每个线程块计算出来的部分和

    seed = 1;
    initMatrix(h_idata, size, seed);   // 初始化数组
    memset(h_odata, 0, gridsize * sizeof(float));

    // CPU归约
    timer.start_cpu();
    float sumOnCPU = ReduceOnCPU(h_idata, size);   // 用单一for循环对所有数组元素进行相加
    timer.stop_cpu();
    std::sprintf(str, "reduce in cpu, result:%f", sumOnCPU);
    timer.duration_cpu<Timer::ms>(str);

    // GPU warmup（GPU在第一次运行时需要的时间较长）
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, blockSize);
    timer.stop_gpu();
    // timer.duration_gpu("reduce in gpu(warmup)");


    // GPU归约(带分支)
    timer.start_gpu();
    ReduceOnGPUWithDivergence(h_idata, h_odata, size, blockSize);   // 见matmul_gpu_basic.cu
    timer.stop_gpu();
    float sumOnGPUWithDivergence = 0;
    for (int i = 0; i < gridsize; i++) sumOnGPUWithDivergence += h_odata[i];   // 在CPU中将GPU计算出来的部分和相加
    std::sprintf(str, "reduce in gpu with divergence, result:%f", sumOnGPUWithDivergence);
    timer.duration_gpu(str);

    // GPU归约(不带分支)
    timer.start_gpu();
    ReduceOnGPUWithoutDivergence(h_idata, h_odata, size, blockSize);   // 见matmul_gpu_basic.cu
    timer.stop_gpu();
    float sumOnGPUWithoutDivergence = 0;
    for (int i = 0; i < gridsize; i++) sumOnGPUWithoutDivergence += h_odata[i];
    std::sprintf(str, "reduce in gpu without divergence, result:%f", sumOnGPUWithoutDivergence);
    timer.duration_gpu(str);
    
    free(h_idata);
    free(h_odata);
    return 0;
}
