/*********************************************************************************************
 * file name  : coresCount.cu
 * author     : 权 双
 * date       : 2023-08-13
 * brief      : 查询GPU计算核心数量
**********************************************************************************************/

/**
 * 可以使用CUDA运行时API：cudaGetDeviceProperties，来查看每个GPU核心的信息
 * 但目前无法通过CUDA运行时API查看GPU的总核心数量，因此需要通过下面这个getSPcores()函数查询
 */

#include <stdio.h>
#include "../tools/common.cuh"

int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
      else printf("Unknown device type\n");
      break;
     case 9: // Hopper
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

int main()
{
    int device_id = 0;   // GPU设备索引号
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);
    

    cudaDeviceProp prop;   // 需要先初始化一个结构体变量作为参数，传递给cudaGetDeviceProperties运行时API
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);   // 注意这里需要传入结构体prop的地址

    printf("Compute cores is %d.\n", getSPcores(prop));

    return 0;
}