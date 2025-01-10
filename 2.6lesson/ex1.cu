#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int id = threadIdx.x + blockIdx.x * blockDim.x; 
    printf("Hello World from block %d and thread %d, global id %d\n", bid, tid, id);
}

/** 
 * 在使用NVCC编译时，可以使用argument -gencode arch=compute_XY,code=sm_XY同时指定多个GPU版本进行编译，例如：
 * nvcc -gencode arch=compute_70,code=sm_70 \
 *    -gencode arch=compute_75,code=sm_75 \
 *    -gencode arch=compute_80,code=sm_80 \
 *    -o output_file source_file.cu
 * 注意：在执行上述指令时，必须要确认实际的CUDA版本支持>=8.0的计算能力
 *
 * 使用NVCC即时编译，可以在编译出的可执行文件中保留PTX文件。需要使用以下指令：
 * nvcc -gencode arch=compute_XY,code=compute_XY，注意：这里的两个计算能力都是compute_XY，即虚拟架构计算能力，
 * 两个架构的计算能力必须保持一致
 * 同时，这个指令也可以简化为：nvcc -arch=sm_XY，等价于：
 * nvcc -gencode arch=compute_XY,code=sm_XY \
 *    -gencode arch=compute_XY,code=sm_XY \
 *
 * 不同版本的CUDA编译器在编译CUDA代码时，都有一个默认的计算能力，例如：CUDA 11.6默认的计算能力是5.2
 * 想要查看默认计算能力，可以使用nvcc helloworld.cu -ptx生成PTX文件，然后打开PTX文件查看.target后标注的默认计算能力
 */
int main(void)
{
    printf("Hello World from CPU!\n");
    hello_from_gpu<<<2, 2>>>();
    cudaDeviceSynchronize();

    return 0;
}