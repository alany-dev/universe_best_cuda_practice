#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
    - 设计算法的时候是按照 Block 设计，写程序的时候按照 thread 写的
    - 索引很容易乱套，所以最好为每一个 Block 设计一个更进一步的索引
    - 绘制一个清晰的 Block 算法图很有必要

*/
#define THREAD_PER_BLOCK 256

__global__ void reduce(float *d_input, float *d_output)
{
    // 提取 block 起始地址 方法
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    int    tid         = threadIdx.x;

    // if(tid == 0 or 2 or 4 or 6)
    //     input_begin[tid] += input_begin[tid + 1];
    // if(tid == 0 or 4)
    //     input_begin[tid] += input_begin[tid + 2];
    // if(tid == 0)
    //     input_begin[tid] += input_begin[tid + 4];

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (tid % (i * 2) == 0) {
            input_begin[tid] += input_begin[tid + i];
        }
        __syncthreads();
    }

    // 不能光写这个。因为 同一个 Block 内的所有线程都会执行这行代码，会引发 线程竞争（写冲突）
    // d_output[blockIdx.x] = input_begin[0];
    if (tid == 0) { // 第0号线程负责拷贝结果
        d_output[blockIdx.x] = input_begin[0];
    }

    // 全局索引 方法
    // int tid = threadIdx.x;
    // int index = blockDim.x * blockIdx.x + tid;
    // for(int i = 1; i < blockDim.x; i *= 2){
    //     if(tid % (2 * i) == 0){
    //         d_input[index] += d_input[index + i];
    //     }
    //     __syncthreads();
    // }
    // if(tid == 0)
    //     d_output[blockDim.x] = d_input[index];
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

    reduce<<<Grid, Block>>>(d_input, d_output);

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