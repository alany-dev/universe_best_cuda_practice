#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
    改进：
        1. 同一个 wrap，没有分支分化

    缺点：
        1. Bank 冲突
            CUDA 共享内存 __shared__ 分成 32 个 Bank
            Bank 0 ~ 31，每个 Bank 一次只能被一个线程访问；如果两个线程同时访问同一个 Bank -> Bank 冲突 -> 速度变慢

        当前的索引公式会让线程在 i≥16 时抢同一个 Bank，所以发生 Bank 冲突。
        tid=0 → index=0   访问  0 和 16
        tid=1 → index=32  访问 32 和 48

        0   %32 = 0
        16  %32 = 16
        32  %32 = 0
        48  %32 = 16

*/

#define THREAD_PER_BLOCK 256

__global__ void reduce2(float *d_input, float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK]; // 必须使用编译期参数，一个Block里共享 共享内存
    // 提取 block 起始地址 方法
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    int    tid         = threadIdx.x;

    shared[tid] = input_begin[tid];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2) {
        if (tid < blockDim.x / (2 * i)) { // 保证每次一半连续的线程同时运行
            /*
                0号线程 ---> 0 1
                1号线程 ---> 2 3
                2号线程 ---> 4 5
                3号线程 ---> 6 7
                tid 号线程 i = 1 ---> tid * 2 * i

                0号线程 ---> 0 2
                1号线程 ---> 4 6
                tid 号线程 i = 2 --> tid * 2 * i
            */
            int index = tid * 2 * i; // 具体是处理哪个下标数字
            shared[index] += shared[index + i];
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

    reduce2<<<Grid, Block>>>(d_input, d_output);

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