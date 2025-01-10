#include <stdio.h>

__global__ void hello_from_gpu()
{
    /**
     * 核函数kernel function必须带有限定词__global__，且返回值必须为void
     * 核函数只能访问GPU显存，不能访问CPU内存
     * 核函数不能使用可变参数，必须一开始就明确参数数量
     * 核函数不能使用静态变量static variables
     * 核函数不能使用函数指针
     * 核函数具有异步性，CPU主机不会等待GPU执行完毕
     */
    printf("Hello World from the the GPU\n");  // 需要包含头文件stdio.h
}


int main(void)
{
    /**
     * CUDA程序编写：
     * 主机代码 - 对GPU进行配置
     * 调用核函数
     * 主机代码 - 将GPU的运算结果回传给主机，并释放内存
     */
    hello_from_gpu<<<1, 1>>>();  // <<<1, 1>>> 用于设置核函数线程，其中第一个数字表示线程块thread block个数，第二个数字表示每个线程块的线程数量
    // 因此，这里只会打印一个Hello World，因为只有一个线程
    cudaDeviceSynchronize();  // 同步CPU与GPU进程

    return 0;
}
