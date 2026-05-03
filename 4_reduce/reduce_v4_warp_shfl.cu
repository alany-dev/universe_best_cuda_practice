#include <chrono> // 用于 CPU 计时
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>


/*
    v0：
        - 每个线程块 有 <BLOCK_SIZE> 1024 个线程
        - shared_mm 记录

        瓶颈：
            - warp 内线程发散
            - 共享内存的 bank conflict
            - 每次 s 迭代都有 __syncthreads()

    v1：
        - 一个线程 global-->shared 读两次，跑满带宽

    v2：
        - 交错寻址，消除取模和大部分的 warp divergence

    v3:
        - shfl
        - 最后 32 个数据的累加，可以不用 __syncthreads()

    v4:
        - warp 内部 shfl 累加，存到 shared_mm[32]
        - shared_mm[32] 内部再次用 shfl 累加
*/

#define FULL_MASK 0xffffffff

__device__ void warpReduce(float *sdata, unsigned int tid)
{
    int var = sdata[tid] + sdata[tid + 32];
    var += __shfl_down_sync(FULL_MASK, var, 16);
    var += __shfl_down_sync(FULL_MASK, var, 8);
    var += __shfl_down_sync(FULL_MASK, var, 4);
    var += __shfl_down_sync(FULL_MASK, var, 2);
    var += __shfl_down_sync(FULL_MASK, var, 1);
    sdata[tid] = var;
}

__inline__ __device__ float block_reduce(float val)
{
    const int        tid      = threadIdx.x;
    const int        warpSize = 32;
    int              lane     = tid % warpSize;
    int              warp_id  = tid / warpSize;
    __shared__ float warpSums[32]; // 一个 Block 最多1024个线程，也就是最多 32 个 warp

    // 每个warp内部都会进行 shuffle
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    if (lane == 0)
        warpSums[warp_id] = val;

    __syncthreads();

    // 由第一个warp 32个线程 处理 shared_mm
    if (warp_id == 0) {
        val = (tid < blockDim.x / warpSize) ? warpSums[tid] : 0.0f;
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduce(float *g_idata, float *g_odata, int n)
{
    float sum = 0.f;
    int   i   = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += gridDim.x * blockDim.x) {
        sum += g_idata[i];
    }

    sum = block_reduce(sum);
    if (threadIdx.x == 0) {
        g_odata[blockIdx.x] = sum;
    }
}

// CPU验证函数
float reduce_cpu(const std::vector<float> &data)
{
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum;
}

int main()
{
    constexpr int BLOCK_SIZE = 1024;
    constexpr int N          = 1024 * 1024; // 1M elements
    constexpr int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::vector<float> h_data(N);

    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f; // 简单起见，全部初始化为1.0
    }
#ifdef PERF_TEST_ENABLED
    // -------------------------------
    // CPU 计时开始
    auto cpu_start = std::chrono::high_resolution_clock::now();
#endif
    float cpu_result = reduce_cpu(h_data);

#ifdef PERF_TEST_ENABLED
    auto                                      cpu_end      = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    // CPU 计时结束
    // -------------------------------

    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;
#endif

    float *d_data, *d_result;
    float *d_final_result;
    float  gpu_result;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, num_blocks * sizeof(float));
    cudaMalloc(&d_final_result, 1 * sizeof(float));

    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

#ifdef PERF_TEST_ENABLED
    // -------------------------------
    // GPU 计时开始 (CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif

    reduce<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    reduce<<<1, num_blocks>>>(d_result, d_final_result, num_blocks);

#ifdef PERF_TEST_ENABLED
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // GPU 计时结束
    // -------------------------------

    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(&gpu_result, d_final_result, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << gpu_result << std::endl;

    if (abs(cpu_result - gpu_result) < 1e-5) {
        std::cout << "Result verified successfully!" << std::endl;
    }
    else {
        std::cout << "Result verification failed!" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    // 清理资源
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_final_result);

    return 0;
}
