#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
Warp 内硬件同步：
    tid < 32 刚好是 1 个 Warp，
    CUDA SIMT 架构保证 Warp 内线程天然同步执行，不需要 __syncthreads()。

减少线程调度开销：
    只有 32 个线程干活，后面的线程直接闲置，
    减少了 warp 调度和指令发射的压力。
*/

#define THREAD_PER_BLOCK 256

__device__ void warpReduce(volatile float *cache, unsigned int tid) // 【改动，volatile】
{
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

__global__ void reduce5(float *d_input, float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK];
    float           *input_begin = d_input + blockDim.x * blockIdx.x * 2; // 【改动三：每个线程处理两倍】
    int              tid         = threadIdx.x;

    shared[tid] =
        input_begin[tid]
        + input_begin[tid + blockDim.x]; // 【改动四：其实就是 在 搬运数据到 共享内存的过程中，每个线程搬运两个数】
    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i /= 2) { // 【改动】
        if (tid < i) {
            shared[tid] += shared[tid + i];
        }
        __syncthreads();
    }

    // 最后 32 个数据，避免if判断，所有线程都累加。【改动】
    // 省略 同步操作。
    /*
        为了禁止 CUDA 编译器对共享内存的「激进优化」，确保在最后 32 个元素的 warp 级操作中，
        每次读写都直接访问共享内存、不被缓存到寄存器、指令不被重排，从而保证结果正确。
    */
    if (tid < 32) {
        warpReduce(shared, tid);
    }

    if (tid == 0) {
        d_output[blockIdx.x] = shared[0];
    }
}


bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++) {
        if (abs(out[i] - res[i]) > 0.005) {
            return false;
        }
    }
    return true;
}

int main()
{
    // printf("hello reduce\n");
    constexpr int N         = 32 * 1024 * 1024;         // 一共要处理的数据量
    int           block_num = N / THREAD_PER_BLOCK / 2; // block数量 【改动一：每个线程处理两个数。】

    float *input  = (float *)malloc(N * sizeof(float)); // input ---> result
    float *result = (float *)malloc(block_num * sizeof(float));

    float *output = (float *)malloc(block_num * sizeof(float)); // d_output ---> output

    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));
    // cpu
    for (int i = 0; i < N; i++) {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // calc
    for (int i = 0; i < block_num; i++) {
        float cur = 0;
        for (int j = 0; j < 2 * THREAD_PER_BLOCK; j++) { // 【改动二：保持 CPU 要处理的数据量】
            cur += input[i * 2 * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

#ifdef PERF_TEST_ENABLED
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif

    reduce5<<<Grid, Block>>>(d_input, d_output);

#ifdef PERF_TEST_ENABLED
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算数据吞吐量 (有效带宽)
    // 逻辑：读取 N 个 float (4bytes)，写出 block_num 个 float
    float data_size_gb   = (N * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
    float time_sec       = milliseconds / 1000.0f;
    float bandwidth_gb_s = data_size_gb / time_sec;

    printf("---------- 性能日志 ----------\n");
    printf("数据规模: %d elements (%.2f MB)\n", N, (N * sizeof(float)) / (1024.0f * 1024.0f));
    printf("核函数耗时: %.3f ms\n", milliseconds);
    printf("有效带宽:   %.2f GB/s\n", bandwidth_gb_s);
    printf("------------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num)) {
        printf("right\n");
    }
    else {
        printf("wrong\n");
        // for(int i = 0 ; i  < block_num; i++){
        //     printf("%lf ", output[i]);
        // }
        // printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);


    return 0;
}