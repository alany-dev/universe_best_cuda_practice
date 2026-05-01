#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
    共享内存
    计算 16*16 的 Block
    把 A 的 16 * K 和 B 的 K * 16 移到共享内存里。

    缺点很明显：
        1. 共享内存使用过多，当前程序没法正常运行
*/

template <unsigned int BLOCK_SIZE, unsigned int BK>
__global__ void sgemm(float *a_ptr, float *b_ptr, float *c_ptr, const int M, const int N, const int K)
{
    int    gx          = blockIdx.x * blockDim.x + threadIdx.x; // 横坐标
    int    gy          = blockIdx.y * blockDim.y + threadIdx.y; // 纵坐标
    int    tx          = threadIdx.x;
    int    ty          = threadIdx.y;
    float *a_ptr_start = a_ptr + blockDim.y * blockIdx.y * K;
    float *b_ptr_start = b_ptr + blockDim.x * blockIdx.x;

    if (gx >= N || gy >= M)
        return;

    __shared__ float a_shared[BLOCK_SIZE][BK];
    __shared__ float b_shared[BK][BLOCK_SIZE];

    /*
        每个线程 负责 gx,gy 相关的数据
        A 16*K
        B 16*K
        每个小块对应位置的数据拷贝
    */
    for (int i = 0; i < K; i += blockDim.x) {
        a_shared[ty][tx] = a_ptr_start[i + ty * K + tx];   // i 每块的起始
        b_shared[ty][tx] = b_ptr_start[(i + ty) * N + tx]; // i*N 每块的起始
    }
    __syncthreads();

    float sum = 0.f;
    for (int i = 0; i < K; i++) {
        sum += a_shared[ty][i] * b_shared[i][tx];
    }
    c_ptr[gy * N + gx] = sum;
}

void random_matrix(int m, int n, float *a)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
#if 1
            a[i * n + j] = 2.0 * (float)drand48() - 1.0;
#else
            a[i * n + j] = (j - i) % 3;
#endif
        }
    }
}

float compare_matrices(int m, int n, float *a, float *b)
{
    int   i, j;
    float max_diff = 0.0f, diff;
    int   printed  = 0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            int idx  = i * n + j;
            diff     = fabs(a[idx] - b[idx]);
            max_diff = (diff > max_diff ? diff : max_diff);
            if (0 == printed) {
                if (max_diff > 1e-4f) {
                    printf("\n error: i %d j %d diff %f got %f expect %f\n", i, j, max_diff, a[idx], b[idx]);
                    printed = 1;
                }
            }
        }
    }
    return max_diff;
}


void cpu_sgemm(float *a_ptr, float *b_ptr, float *c_ptr, const int M, const int N, const int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int t = 0; t < K; t++) {
                sum += a_ptr[i * K + t] * b_ptr[t * N + j];
            }
            c_ptr[i * N + j] = sum;
        }
    }
}


int main()
{
    constexpr int    m          = 512;
    constexpr int    n          = 512;
    constexpr int    k          = 512;
    constexpr size_t mem_size_a = m * k * sizeof(float);
    constexpr size_t mem_size_b = n * k * sizeof(float);
    constexpr size_t mem_size_c = m * n * sizeof(float);

    float *matrix_a_host          = (float *)malloc(mem_size_a);
    float *matrix_b_host          = (float *)malloc(mem_size_b);
    float *matrix_c_host_cpu_calc = (float *)malloc(mem_size_c);
    float *matrix_c_host_gpu_calc = (float *)malloc(mem_size_c);

    random_matrix(m, k, matrix_a_host);
    random_matrix(k, n, matrix_b_host);
    memset(matrix_c_host_cpu_calc, 0, mem_size_c);
    memset(matrix_c_host_gpu_calc, 0, mem_size_c);

    float *matrix_a_device;
    float *matrix_b_device;
    float *matrix_c_device;
    cudaMalloc((void **)&matrix_a_device, mem_size_a);
    cudaMalloc((void **)&matrix_b_device, mem_size_b);
    cudaMalloc((void **)&matrix_c_device, mem_size_c);

    cudaMemcpy(matrix_a_device, matrix_a_host, mem_size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_b_device, matrix_b_host, mem_size_b, cudaMemcpyHostToDevice);

    // cpu calc
    cpu_sgemm(matrix_a_host, matrix_b_host, matrix_c_host_cpu_calc, m, n, k);

    // 最普通的方法，每个线程都负责一个 m*n 的数据
    constexpr int BLOCK = 16;
    dim3          Grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);
    dim3          Block(BLOCK, BLOCK);

#ifdef PERF_TEST_ENABLED
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif

    // 运行不了的
    // sgemm<BLOCK, k><<<Grid, Block>>>(matrix_a_device, matrix_b_device, matrix_c_device, m, n, k);

#ifdef PERF_TEST_ENABLED
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算数据吞吐量 (有效带宽)
    // 逻辑：读取 N 个 float (4bytes)，写出 block_num 个 float
    float data_size_gb   = ((float)m * k + (float)k * n + (float)m * n) * sizeof(float) / (1024.0f * 1024.0f * 1024.0f);
    float flops          = 2.0f * (float)m * (float)n * (float)k;
    float time_sec       = milliseconds / 1000.0f;
    float bandwidth_gb_s = data_size_gb / time_sec;
    float tflops         = flops / time_sec / 1e12;

    printf("---------- 性能日志 ----------\n");
    printf("矩阵规模: M=%d, N=%d, K=%d\n", m, n, k);
    printf("核函数耗时: %.3f ms\n", milliseconds);
    printf("有效带宽:   %.2f GB/s\n", bandwidth_gb_s);
    printf("计算性能:   %.3f TFLOPS\n", tflops);
    printf("------------------------------\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    cudaMemcpy(matrix_c_host_gpu_calc, matrix_c_device, mem_size_c, cudaMemcpyDeviceToHost);

    float max_err = compare_matrices(m, n, matrix_c_host_gpu_calc, matrix_c_host_cpu_calc);
    printf("Max error: %f\n", max_err);

    free(matrix_a_host);
    free(matrix_b_host);
    free(matrix_c_host_cpu_calc);
    free(matrix_c_host_gpu_calc);

    cudaFree(matrix_a_device);
    cudaFree(matrix_b_device);
    cudaFree(matrix_c_device);

    return 0;
}