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

    v1 优化
        - 共享内存

        瓶颈：
            - 共享内存的 bank conflict：
                CUDA 共享内存被划分为 32 个 bank（每个 bank 4 字节宽度），
                同一 warp 内的多个线程若访问同一 bank 的不同地址就会发生 bank conflict，导致访存串行化。

                BDIMX == BDIMY
                同一个 warp 的 32 个线程写回数据的时候（threadIdx.x 连续，irow = threadIdx.y 固定，icol = threadIdx.x）
                因此该 warp 的 32 个线程同时读取 tile[0][irow], tile[1][irow], …, tile[31][irow]。
                    tile[i][j] 地址 = base + (i * BDIMX + j) * 4

                    当 BDIMX = 32 时，行步长为 32 个 float，即 i * 32 * 4。

                    所有线程的地址除以 4 后对 32 取模，(base/4 + i*32 + irow) % 32 = (base/4 + irow) %
   32，全部落在同一个 bank。 → 32 路 bank conflict，共享内存带宽下降约 32 倍。


            在 Step A (读入) 时： 线程 ID (tx, ty) 负责坐标 (tx, ty)。
            在 Step B (写出) 时： 我们让线程 ID (tx, ty) 去负责目标坐标 (irow, icol) 。

    v2 优化：
        - pad = 1
        - 共享内存数组的列数从 BDIMX --> BDIMX +1
        - 以 BDIMX = BDIMY = 32 为例。同一 warp 中，threadIdx.x 连续（0 ~ 31），而 threadIdx.y 相同，因此：
            irow = bidx / 32，当 threadIdx.y 固定时，irow 也是固定的（等于 threadIdx.y）。
            icol = bidx % 32，随着 threadIdx.x 连续变化，icol 也就从 0 变化到 31。

        // tile[0][irow + 1] t0
        // tile[1][irow + 1] t1
        // tile[2][irow + 1] t2

        // 每个线程之间访问的字节差为(irow + 1) × sizeof(float) = 132字节
        // t0   0     bank0
        // t1   132   bank1
        // t2   264   bank2

    v3 优化：
        - 每个线程 处理两个元素
*/
template <unsigned int BDIMX, unsigned int BDIMY> __global__ void transpose(float *out, float *in, int nx, int ny)
{
    constexpr int    IPAD = 1;
    __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

    // 原数据坐标
    unsigned int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // 原数据在 in 的索引下标
    unsigned int ti = iy * nx + ix;
    // 当前线程在线程块 32*32 内部的 索引
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    // 转置到 out 位置，在 blockDim.x 行 blockDim.y 列的 坐标。
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // 目标 out，是 nx 行 ny 列。
    ix = blockIdx.y * blockDim.y + icol;
    iy = 2 * blockIdx.x * blockDim.x + irow;

    unsigned int to = iy * ny + ix;

    if (ix < nx && iy < ny) {
        unsigned int row_idx  = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
        tile[row_idx]         = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];

        __syncthreads();

        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to]              = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

// 调用核函数的封装函数
void call_transpose(float *d_out, float *d_in, int nx, int ny)
{
    constexpr int BDIMX = 32;
    constexpr int BDIMY = 32;
    dim3          blockSize(BDIMX, BDIMY); // 线程块大小
    auto          grid = (nx + BDIMX - 1) / BDIMX;
    dim3          gridSize(int(grid / 2), (ny + BDIMY - 1) / BDIMY);
    transpose<BDIMX, BDIMY><<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
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