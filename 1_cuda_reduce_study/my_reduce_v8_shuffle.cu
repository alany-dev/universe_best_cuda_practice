#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/*
    1. shuffle 只能再同一个 wrap 内进行操作

    __shfl_down_sync 完全依赖 Warp 的 硬件锁补 Lockstep 执行
    通过 mask 参数强制要求参与的线程必须同时到达这条指令
    同一个 Warp 内的线程会同步执行这条指令，读取的是彼此执行前的寄存器值，执行完后同步进入下一条指令，天然一致！
*/

#define THREAD_PER_BLOCK 256

template<unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce8(float  *d_input, float* d_output)
{
    int tid = threadIdx.x;
    float sum = 0.f;
    
    float *input_begin = d_input + NUM_PER_BLOCK * blockIdx.x; 

    for(int i = 0 ; i < NUM_PER_THREAD ; i++){
        sum += input_begin[tid + i * THREAD_PER_BLOCK];
    }
    // sum 会把 当前线程负责的 所有全局内存数据，累加完毕。
    // 当前 Block 里，就会剩下 thread_size 个 线程的 sum。

    // 当前 Block 里，每一个 warp(32个线程) 进行 shuffle，最终 每个 warp 的 第 0 号线程存有 warp 的累加和 (sum)
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1); 

    __shared__ float warpLevelSums[32];
    const int laneId = tid % 32;
    const int warpId = tid / 32;
    
    if(laneId == 0){
        warpLevelSums[warpId] = sum;
    }

    __syncthreads();

    if(warpId == 0){
        // 保证 当前第一个warp的32个线程，都持有了对应的sum数据
        sum = (laneId < blockDim.x / 32) ? warpLevelSums[laneId] : 0.f;    
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1); 
    }
    
    if(tid == 0){
        d_output[blockIdx.x] = sum;
    }
}


bool check(float* out, float *res, int n){
    for(int i = 0 ; i < n; i++){
        if(abs(out[i] - res[i]) > 0.005){
            return false;
        }
    }
    return true;
}

int main()
{
    // printf("hello reduce\n");
    constexpr int N = 32 * 1024 * 1024;  // 一共要处理的数据量
    constexpr int block_num = 1024; // block数量 【改动】
    constexpr int num_per_block = N / block_num;  // 每个 Block 要处理的数据量
    constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;    // 每个 Thread 要处理的数据量

    float *input = (float*)malloc(N * sizeof(float));   // input ---> result
    float *result = (float*)malloc(block_num * sizeof(float));

    float *output = (float*)malloc(block_num * sizeof(float));  // d_output ---> output

    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));
    // cpu
    for(int i = 0 ; i < N; i++){
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // calc
    for(int i = 0; i < block_num; i++){
        float cur = 0;
        for(int j = 0 ; j < num_per_block; j++){     // 【改动】
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

    reduce8<num_per_block, num_per_thread><<<Grid, Block>>>(d_input, d_output);

#ifdef PERF_TEST_ENABLED
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop); 

    // 计算数据吞吐量 (有效带宽)
    // 逻辑：读取 N 个 float (4bytes)，写出 block_num 个 float
    float data_size_gb = (N * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
    float time_sec = milliseconds / 1000.0f;
    float bandwidth_gb_s = data_size_gb / time_sec;

    printf("---------- 性能日志 ----------\n");
    printf("数据规模: %d elements (%.2f MB)\n", N, (N*sizeof(float))/(1024.0f*1024.0f));
    printf("核函数耗时: %.3f ms\n", milliseconds);
    printf("有效带宽:   %.2f GB/s\n", bandwidth_gb_s);
    printf("------------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if(check(output, result, block_num)){
        printf("right\n");
    }else{
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