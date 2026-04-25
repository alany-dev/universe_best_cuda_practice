#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
    1. 限定 Block 大小 1024
    2. shared 共享内存默认是 THREAD_PER_BLOCK，为了 线程 --- 共享内存，一一对应
    3. global ---> shared 需要，让每个Block线程先进行累加合并
*/

#define THREAD_PER_BLOCK 256

__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce7(float *d_input, float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK];
    int              tid = threadIdx.x;

    // float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    // shared[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    // __syncthreads();


    float *input_begin = d_input + NUM_PER_BLOCK * blockIdx.x; // 【改动】
    shared[tid]        = 0;
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        shared[tid] += input_begin[tid + i * THREAD_PER_BLOCK];
    }
    __syncthreads();

#pragma unroll
    for (int i = blockDim.x / 2; i > 32; i /= 2) {
        if (tid < i) {
            shared[tid] += shared[tid + i];
        }
        __syncthreads();
    }

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
    constexpr int N              = 32 * 1024 * 1024;                 // 一共要处理的数据量
    constexpr int block_num      = 1024;                             // block数量 【改动】
    constexpr int num_per_block  = N / block_num;                    // 每个 Block 要处理的数据量
    constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK; // 每个 Thread 要处理的数据量

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
        for (int j = 0; j < num_per_block; j++) { // 【改动】
            cur += input[i * num_per_block + j];
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

    reduce7<num_per_block, num_per_thread><<<Grid, Block>>>(d_input, d_output);

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