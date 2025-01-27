#include "cuda_runtime.h"
#include "cuda.h"
#include "stdio.h"
#include "utils.hpp"

// 注意这里d_idata和d_odata分别为输入数据和输出结果在GPU显存中的指针。访问GPU显存中的数组数据必须通过指针实现。
__global__ void ReduceNeighboredWithDivergence(float *d_idata, float *d_odata, int size){
    // set thread ID
    unsigned int tid = threadIdx.x;   // 线程在线程块中的索引
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 线程的全局索引

    // convert global data pointer to the local pointer of this block
    float *idata = d_idata + blockIdx.x * blockDim.x;   // 当前线程块在GPU显存中的全局初始地址

    // boundary check
    if (idx >= size) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // 第一轮：只有线程在线程块中的索引为2的倍数时时，才进行相加的操作
        // 第二轮：只有线程在线程块中的索引为4的倍数时时，才进行相加的操作，以此类推
        // 这样会造成严重的线程束分化
        if ((tid % (2 * stride)) == 0)
        {
            /**
             * 在C/C++中，数组名本质上是一个指针。例如，float arr[10]，这里的数组名arr本质上是指向数组第一个元素
             * 的指针，即arr等价于&arr[0]。因此，arr[i]也等价于*(arr + i)。
             * 因此，指针索引可以像数组索引一样使用。这里的idata[tid]也就等价于*(idata + tid)
             */
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();   // 在进行下一轮操作时，线程块中的所有元素都必须计算完成（必须先计算出所有部分和，才能进行下一步）
    }

    // write result for this block to global mem
    if (tid == 0) d_odata[blockIdx.x] = idata[0];   // 将计算完成后的部分和idata[0]传递给d_odata，仅使用0号线程完成这个操作
}

__global__ void ReduceNeighboredWithoutDivergence(float *d_idata, float *d_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = d_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        /**
         * 注：在下面的例子中，线程索引即为前面的tid = threadIdx.x;
         * 假设 blockDim.x = 8，初始数据为 [1, 2, 3, 4, 5, 6, 7, 8]，归约过程如下：
         * stride = 1：
         * index = 2 * 1 * tid，即 index = 0, 2, 4, 6。
         * 线程 0 处理 idata[0] += idata[1]，结果 [3, 2, 7, 4, 11, 6, 15, 8]。
         * 线程 1 处理 idata[2] += idata[3]。
         * 线程 2 处理 idata[4] += idata[5]。
         * 线程 3 处理 idata[6] += idata[7]。

         * stride = 2：
         * index = 2 * 2 * tid，即 index = 0, 4。
         * 线程 0 处理 idata[0] += idata[2]，结果 [10, 2, 7, 4, 26, 6, 15, 8]。
         * 线程 1 处理 idata[4] += idata[6]。

         * stride = 4：
         * index = 2 * 4 * tid，即 index = 0。
         * 线程 0 处理 idata[0] += idata[4]，结果 [36, 2, 7, 4, 26, 6, 15, 8]。
         * 最终，idata[0] 包含了整个线程块的归约结果。
         *
         * 通过以上操作，我们确保了同一个线程束中的所有线程执行相同的代码路径，从而避免了线程束分化。
         * 如果只考虑核函数的执行速度（不考虑内存访问），避免线程束分化可以大大提升核函数的执行效率。
         */
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) d_odata[blockIdx.x] = idata[0];
}


void ReduceOnGPUWithDivergence(float *h_idata, float *h_odata, int size, int blockSize)
{
    int ibytes = size * sizeof(float);
    int obytes = size / blockSize * sizeof(float);

    memset(h_odata, 0, obytes);

    float* d_idata = nullptr;
    float* d_odata = nullptr;

    CUDA_CHECK(cudaMalloc(&d_idata, ibytes));   // 在GPU上开辟内存
    CUDA_CHECK(cudaMalloc(&d_odata, obytes));

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));   // 将CPU上的数据拷贝到GPU上
   
    dim3 block(blockSize);
    dim3 grid(size / blockSize);
    ReduceNeighboredWithDivergence <<<grid, block>>> (d_idata, d_odata, size);   // 运行核函数

    // 将结果从device拷贝回host
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    //注意在同步后，检测核函数
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaFree(d_odata));   // 释放GPU内存
    CUDA_CHECK(cudaFree(d_idata));
}

void ReduceOnGPUWithoutDivergence(float *h_idata, float *h_odata, int size, int blockSize)
{
    int ibytes = size * sizeof(float);
    int obytes = size / blockSize * sizeof(float);

    memset(h_odata, 0, obytes);
    
    float* d_idata = nullptr;
    float* d_odata = nullptr;

    CUDA_CHECK(cudaMalloc(&d_idata, ibytes));
    CUDA_CHECK(cudaMalloc(&d_odata, obytes));

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, ibytes, cudaMemcpyHostToDevice));
    
    dim3 block(blockSize);
    dim3 grid(size / blockSize);
    ReduceNeighboredWithoutDivergence <<<grid, block>>> (d_idata, d_odata, size);

    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, obytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaFree(d_odata));
    CUDA_CHECK(cudaFree(d_idata));
}

