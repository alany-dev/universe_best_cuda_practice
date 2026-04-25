#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
    1 个 SM 的 Shared Memory 固定划分为 32 个 Bank
    每个 Bank 的基础访问粒度 = 4 Byte（32bit）

    行 \ 列	    Bank0	    Bank1	    Bank2	…	Bank31
    第 0 行	    s[0]	    s[1]	    s[2]	…	s[31]
    第 1 行	    s[32]	    s[33]	    s[34]	…	s[63]
    第 2 行	    s[64]	    s[65]	    s[66]	…	s[95]
    …	        跨行循环

    改进：
        1. Bank 冲突


*/

#define THREAD_PER_BLOCK 256

__global__ void reduce3(float *d_input, float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK]; // 必须使用编译期参数，一个Block里共享 共享内存
    float           *input_begin = d_input + blockDim.x * blockIdx.x;
    int              tid         = threadIdx.x;

    shared[tid] = input_begin[tid];
    __syncthreads();

    /*  0 1 2 3 4 5 6 7

        0号线程 ---> 0 4
        1号线程 ---> 1 5
        2号线程 ---> 2 6
        3号线程 ---> 3 7
        tid 号线程 i = 4 ---> tid  tid + i

        0号线程 ---> 0 2
        1号线程 ---> 1 3
        tid 号线程 i = 2 --> tid  tid + i
    */
    for (int i = blockDim.x / 2; i >= 1; i /= 2) {
        if (tid < i) {
            // 每个线程 只访问 同一列（同一个 Bank）
            shared[tid] += shared[tid + i];
        }
        __syncthreads();
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
    constexpr int N         = 32 * 1024 * 1024;     // 一共要处理的数据量
    int           block_num = N / THREAD_PER_BLOCK; // block数量

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
        for (int j = 0; j < THREAD_PER_BLOCK; j++) {
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(N / THREAD_PER_BLOCK, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

#ifdef PERF_TEST_ENABLED
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif

    reduce3<<<Grid, Block>>>(d_input, d_output);

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