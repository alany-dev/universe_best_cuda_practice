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
*/
template <unsigned int BLOCK_SIZE> __global__ void reduce(float *g_idata, float *g_odata)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
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
    constexpr int N          = 1024 * 1024;                           // 1M elements
    constexpr int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2; // 减半


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

    reduce<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(d_data, d_result);
    reduce<BLOCK_SIZE><<<1, num_blocks>>>(d_result, d_final_result);

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
