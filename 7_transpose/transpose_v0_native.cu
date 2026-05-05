#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

/*
    v0: 原生方法，每个线程负责一个数据的转置
        瓶颈：
            - 32 x 32 二维线程块，一个 warp 的 32 个线程， ix 连续，iy 相同
            - 读是完全合并的：一个 128 字节内存事务可以一次性服务整个 warp
            - 写是非合并的：每个线程写入的地址间隔 ny * sizeof(float) 字节，32 个线程跨越 32 * ny * 4 字节

*/
__global__ void transpose(float *out, float *in, int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// 调用核函数的封装函数
void call_transpose(float *d_out, float *d_in, int nx, int ny)
{
    dim3 blockSize(32, 32); // 线程块大小
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);
    transpose<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

int main()
{
    int    nx   = 4096;
    int    ny   = 4096;
    size_t size = nx * ny * sizeof(float);

    // 主机内存分配
    float *h_in      = (float *)malloc(size);
    float *h_out     = (float *)malloc(size);
    float *h_cpu_out = (float *)malloc(size); // 用于CPU转置结果

    // 初始化输入矩阵
    for (int i = 0; i < nx * ny; i++) {
        h_in[i] = float(int(i) % 11);
    }

    // 设备内存分配
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // GPU预热
    int warm_up_iter = 5;
    for (int i = 0; i < warm_up_iter; ++i) {
        call_transpose(d_out, d_in, nx, ny);
    }

    int bench_iter = 5;
    // 开始计时 GPU
    cudaEventRecord(start);

    for (int i = 0; i < bench_iter; ++i) {
        call_transpose(d_out, d_in, nx, ny);
    }

    // 结束计时
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        // 释放已分配资源后退出
        free(h_in);
        free(h_out);
        free(h_cpu_out);
        cudaFree(d_in);
        cudaFree(d_out);
        return -1;
    }

    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    float gpu_avg_ms = gpu_milliseconds / float(bench_iter);
    std::cout << "GPU naive transpose average time: " << gpu_avg_ms << " ms" << std::endl;

    // 将 GPU 结果复制回主机
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // ------- CPU 转置与计时 -------
    // CPU 预热
    for (int iter = 0; iter < warm_up_iter; ++iter) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                h_cpu_out[i * ny + j] = h_in[j * nx + i];
            }
        }
    }

    // CPU 正式计时
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < bench_iter; ++iter) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                h_cpu_out[i * ny + j] = h_in[j * nx + i];
            }
        }
    }
    auto                                      cpu_end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_elapsed = cpu_end - cpu_start;
    double                                    cpu_avg_ms  = cpu_elapsed.count() / bench_iter;
    std::cout << "CPU transpose average time: " << cpu_avg_ms << " ms" << std::endl;

    // ------- 结果验证 -------
    float max_error = 0.0f;
    for (int i = 0; i < nx * ny; ++i) {
        float diff = std::fabs(h_out[i] - h_cpu_out[i]);
        max_error  = std::max(max_error, diff);
    }
    std::cout << "Max error between GPU and CPU: " << max_error << std::endl;

    // 加速比
    if (gpu_avg_ms > 0.0f) {
        std::cout << "Speedup (CPU / GPU): " << cpu_avg_ms / gpu_avg_ms << "x" << std::endl;
    }
    else {
        std::cout << "GPU time too small to compute speedup." << std::endl;
    }

    // 释放内存
    free(h_in);
    free(h_out);
    free(h_cpu_out);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Matrix transposition completed successfully." << std::endl;
    return 0;
}