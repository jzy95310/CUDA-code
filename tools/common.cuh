#pragma once
#include <stdlib.h>
#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    /**
     * error_code为CUDA运行时返回的错误代码（cudaSuccess = 0），filename参数一般为__FILE__，即发生cudaError的文件名称，
     * lindeNumber参数一般为__LINE__，即在文件中发生cudaError的具体行数
     * 这里，cudaGetErrorName和cudaGetErrorString都为CUDA运行时API，分别返回cudaError的名称和cudaError的具体信息，返回值都为string
     */
    if (error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
                error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}



void setGPU()
{
    // 检测计算机GPU数量
    int iDeviceCount = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount), __FILE__, __LINE__);

    if (error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else
    {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }
    // 设置执行
    int iDev = 0;
    error = ErrorCheck(cudaSetDevice(iDev), __FILE__, __LINE__);
    if (error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for computing.\n");
    }
}

/************************* 未包含错误检测代码 ******************************************************************************
void setGPU()
{
    // 检测计算机GPU数量
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

    if (error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else
    {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }
    // 设置执行
    int iDev = 0;
    error = cudaSetDevice(iDev);
    if (error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for computing.\n");
    }
}
*********************************************************************************************************************/

