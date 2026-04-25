#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
    让单个线程做更多的事~

    改进 Plan A：
        1. 减少 block 梳理
        2. 保持 block 中 thread 的梳理
        3. 让每个 thread 处理更多的数据

    > 其实就是 在 搬运数据到 共享内存的过程中，每个线程搬运两个数

    Q 为什么 2 个最好，3、4 个反而不行？
    A   1. GPU 全局内存【只能按 128 字节读取】
            显存一次读取 = 32 个 float（128B）刚好是 1 个 warp（32 线程） 一次读取的量
            每个线程读 2 个：
            32 线程 × 2 = 64 个 float→ 刚好 2 个显存事务，完美对齐！
        2. 寄存器数量有限
            你读越多：
            要存的临时变量越多
            寄存器占用越多
            GPU 能同时运行的 Block 越少
            最终并行度下降 → 速度变慢
        3. 最关键：加法不能无限合并
*/

#define THREAD_PER_BLOCK 256

__global__ void reduce4_plan_a(float *d_input, float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK];
    float           *input_begin = d_input + blockDim.x * blockIdx.x * 2; // 【改动三：每个线程处理两倍】
    int              tid         = threadIdx.x;

    shared[tid] =
        input_begin[tid]
        + input_begin[tid + blockDim.x]; // 【改动四：其实就是 在 搬运数据到 共享内存的过程中，每个线程搬运两个数】
    __syncthreads();

    for (int i = blockDim.x / 2; i >= 1; i /= 2) {
        if (tid < i) {
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

    reduce4_plan_a<<<Grid, Block>>>(d_input, d_output);

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