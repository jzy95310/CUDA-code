#include <stdio.h>


int main(void)
{
    // 检测计算机GPU数量
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);   // cudaGetDeviceCount可以在主机或设备函数中调用

    if (error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else
    {
        // 当返回变量为cudaSuccess时，说明cudaGetDeviceCount函数调用成功了
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }
    
    // 设置执行
    int iDev = 0;  // 想要使用的GPU的索引号
    error = cudaSetDevice(iDev);   // cudaSetDevice只能在主机函数中调用
    if (error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for computing.\n");
    }

    return 0;
}

