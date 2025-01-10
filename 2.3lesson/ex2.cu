#include <stdio.h>

/**
 * 一维网格：
 * grid_size, block_size保存在内建变量（built-in variables）中，其中 gridDim.x = grid_size, blockDim.x = block_size
 * 每个线程都有一个独立的thread index，可通过blockIdx.x和threadIdx.x确定
 * 其中blockIdx.x为线程块在当前网格中的索引值, threadIdx.x为线程在当前线程块中的索引值
 * 因此线程索引 = threadIdx.x + blockIdx.x * blockDim.x
 *
 * 二/三维网格：
 * 定义多维网格时，可以使用C++的struct结构：
 * dim3 grid_size(Gx,Gy,Gz); 其中Gy, Gz分别对应gridDim.y, gridDim.z
 * dim3 block_size(Bx,By,Bz); 其中By, Bz分别对应blockDim.y, blockDim.z
 * 注意：多维网格和线程块在本质上是一维的，物理意义上不分块
 * 以三维网格三维线程块为例，可以通过如下方法计算索引：
 * 当前线程块中的线程索引：int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
 * 当前网格中的线程块索引：int bid = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
 * 全局线程索引：int id = bid * (blockDim.x * blockDim.y + blockDim.z) + tid
 */

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;   // 0 <= blockIdx.x < 2
    const int tid = threadIdx.x;   // 0 <= threadIdx.x < 4

    const int id = threadIdx.x + blockIdx.x * blockDim.x;   // 0 <= id < 8
    printf("Hello World from block %d and thread %d, global id %d\n", bid, tid, id);
}


int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}

/**
 * 在使用NVCC编译时，NVCC会将源代码为两部分：主机代码（Host）与设备代码（Device）
 * NVCC先将设备代码编译为PTX伪汇编代码，此时需要用argument -arch=compute_XY指定一个虚拟架构的计算能力，应尽量低，以适配更多实际GPU
 * 随后，NVCC再将PTX代码编译为二进制的cubin目标代码，此时需要用argument -code=sm_XY指定一个真实架构的计算能力，应尽量高，以发挥GPU最大性能
 * 全部的arguments可以参考https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#nvcc-command-options
 * 注意：真实架构的计算能力必须大于虚拟架构的计算能力
 * 
 * 每款GPU都有用于标识的“计算能力”（即版本号），通常用X.Y表示，其中X为主版本号，Y为次版本号，不同主版本之间通常不能兼容
 * 在使用NVCC编译.cu源码为PTX代码时，可以使用-arch=compute_XY指定虚拟架构的计算能力，其中X为主版本号，Y为次版本号
 * 例如，使用nvcc helloworld.cu -o helloworld -arch=compute_61编译出的可执行文件只能在cuda版本号>=6.1的GPU上执行
 * 在将PTX代码转化为cubin二进制代码时，可以使用-code=sm_XY只能真实架构的计算能力，其中X为主版本号，Y为次版本号
 * 例如，使用nvcc helloworld.cu -o helloworld -arch=compute_61 -code=sm_61
 * 注意：
 * 一旦使用了-code flag指定了真实架构计算能力，则必须用-arch flag来指定虚拟架构计算能力，但反之则不需要
 * 二进制cubin代码大版本之间不兼容（例如6.1和7.0之间不兼容）
 * 真实架构计算能力需要<=实际GPU的计算能力
 * 
 * GPU的综合性能通常包括显存容量、显存带宽、单/双精度浮点数运算峰值（通常以GFLOPS或TFLOPS为单位）
 */