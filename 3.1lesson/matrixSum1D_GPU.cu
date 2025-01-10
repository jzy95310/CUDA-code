/*********************************************************************************************
 * file name  : matrixSum1D_GPU.cu
 * author     : 权 双
 * date       : 2023-08-04
 * brief      : 矩阵求和程序，通过调用核函数在GPU执行
***********************************************************************************************/

/**
 * CUDA通过内存分配、数据传递、内存初始化、内存释放进行内存管理，分别对应以下的CUDA内存管理函数：
 * 内存分配 - cudaMalloc：在GPU的内存中分配一段地址
 * 数据传递 - cudaMemcpy：在主机与设备之间进行数据与数据的交换和拷贝
 * 内存初始化 - cudaMemset：在GPU的内存中分配地址的同时传入一段数据
 * 内存释放 - cudaFree：内存使用完毕后，无论是主机还是设备，都需要释放占用的内存空间
 */

/**
 * cudaMalloc在主机和设备中都可以进行调用。它默认返回一个cudaError的错误代码。
 * cudaMalloc有两个输入参数：void **devPtr和size_t size，其中**devPtr为一个双重指针，它指向所要分配的内存地址
 * 而size为所需要分配的内存大小，单位为bytes
 *
 * cudaMemcpy只能在主机函数中调用。它默认返回一个cudaError的错误代码。
 * cudaMemcpy有四个输入参数：void *dst, void *src, size_t count, cudaMemcpyKind kind，
 * 分别为目标地址，原地址，拷贝字节数，和拷贝方向（cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, 
 * cudaMemcpyDeviceToDevice, 以及cudaMemcpyDefault）
 * 当设置为默认值cudaMemcpyDefault，则根据实际传入的数据来判断拷贝方向，但只允许在支持统一虚拟寻址的系统中使用
 *
 * cudaMemset只能在主机函数中调用。它默认返回一个cudaError的错误代码。
 * cudaMemset有三个输入参数：void *devPtr, int value, size_t count
 * 分别为需要初始化的内存地址，初始化值（例如0），以及初始化内存的字节数。初始化的目的是为了防止程序错误地访问未初始化过的内存地址
 *
 * cudaFree在主机和设备中都可以进行调用。它默认返回一个cudaError的错误代码。
 * cudaFree只有一个输入参数：void *devPtr，为需要释放的内存地址
 */

#include <stdio.h>
#include "../tools/common.cuh"

// 注：这里定义核函数时，用的修饰符为__global__，且核函数只能返回void
__global__ void addFromGPU(float *A, float *B, float *C, const int N)
{
    // 相比于在CPU中的计算，这里省去了一个for循环
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x; 

    C[id] = A[id] + B[id];
    
}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        // rand()为[0, RAND_MAX)之间的随机数，0xFF即为255，或二进制的11111111，这里rand() & 0xFF即随机生成一个不大于255的随机数
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main(void)
{
    // 1、设置GPU设备
    setGPU();

    // 2、分配主机内存和设备内存，并初始化
    int iElemCount = 512;                               // 设置元素数量
    size_t stBytesCount = iElemCount * sizeof(float);   // 字节数
    
    // （1）分配主机内存，并初始化
    float *fpHost_A, *fpHost_B, *fpHost_C;   // 初始化3个指针，用于存储内存分配的返回值
    fpHost_A = (float *)malloc(stBytesCount);   // malloc为C语言中默认的分配内存所调用的函数
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)   // 若3个指针都不是空指针，说明内存分配成功
    {
        memset(fpHost_A, 0, stBytesCount);  // 主机内存初始化为0
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    // （2）分配设备内存，并初始化
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;   // 初始化3个指针
    cudaMalloc((float**)&fpDevice_A, stBytesCount);   // 注意这里的(float**)&fpDevice_A为双重指针
    cudaMalloc((float**)&fpDevice_B, stBytesCount);
    cudaMalloc((float**)&fpDevice_C, stBytesCount);
    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        cudaMemset(fpDevice_A, 0, stBytesCount);  // 设备内存初始化为0
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
    }
    else
    {
        // 如果发生错误，则需要释放之前初始化的主机内存
        printf("fail to allocate memory\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    // 3、初始化主机中数据
    srand(666); // 设置随机种子
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);
    
    // 4、数据从主机复制到设备
    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice); 
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice); 
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);


    // 5、调用核函数在设备中进行计算
    dim3 block(32);   // 每个线程块大小为32
    dim3 grid(iElemCount / 32);   // 网格大小为：512 / 32 = 16，这样可以保证每一个元素的计算都对应一个线程

    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);    // 调用核函数
    // cudaDeviceSynchronize();   // 主机等待核函数计算完毕，但这里可以省略，因为下一步cudaMemcpy带有一个隐式的同步

    // 6、将计算得到的数据从设备传给主机
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);


    for (int i = 0; i < 10; i++)    // 打印
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    // 7、释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();
    return 0;
}

/**
 * 主机函数（即带有__host__修饰符，一般可省略）用于定义在主机上运行的函数。如果一个函数可以同时在主机与设备被调用，
 * 则可以用__host__和__device__同时修饰该函数
 * 设备函数（即带有__device__修饰符）只能被核函数或其他设备函数调用，用于定义只能在GPU设备上运行的函数。
 * 在定义核函数时，它的修饰符为__global__，一般由主机调用，但在设备上执行
 * 注意：__global__不能和__host__或__device__同时使用
 */