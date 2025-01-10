#include <stdio.h>

/**
 * 每个核函数启动所产生的所有线程统称为一个网格（grid）
 * 每个网格中有若干个线程块（block），每个线程块中包含若干个线程（thread）。线程是GPU编程中的最小单位。
 * 每个网格大小最多不能超过2^31 - 1，每个线程块大小最多不能超过1024
 * 总线程个数（grid_size * block_size）可远大于GPU的数量，想充分利用GPU的计算资源，至少需要保证总线程数 >= GPU数量
 */

__global__ void hello_from_gpu()
{
    printf("Hello World from the the GPU\n");
}


int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}