#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/*
    共享内存
    计算 16*16 的 Block
    把 A 的 16 * K 和 B 的 K * 16 移到共享内存里。
    此时，改成 每次把 A 的 16*16 和 B 的 16*16 放到共享内存里，
    每个线程会用局部变量记录 1*16 x 16*1 的 1*1 结果，最后累加即可得到。

    一个线程负责四个数的处理。
        - Block 的数量 变成原本的 1/4
        - Block 线程块里的线程数 不变 16 * 16

        此时 一个 Block 里的线程 要 处理 原本四个Block 的 数据。

        共享内存 x4，一次性存原本 4个 Block 的数据量。

    v3 版本是 一个线程，Block 16*16，处理 32*32 的数据。四个离散的数。

    v4 版本改成
        一个 Block 线程块，32*8。同样要处理 32 * 32 的数据，四个连续的数。

*/

// 这么写，好处是 pointer 意图接受一个左值，而不仅仅是一个指针
// 取地址 -> 强转 -> 下标访问
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// 其实 这里的 BM == blckDim.y  BN == blockDim.x
template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int NUM_PER_THREAD>
__global__ void sgemm(float *a_ptr, float *b_ptr, float *c_ptr, const int M, const int N, const int K)
{
    int tx = threadIdx.x; // 0-8
    int ty = threadIdx.y; // 0-31
    int bx = blockIdx.x;
    int by = blockIdx.y;
    a_ptr  = &a_ptr[by * BM * K];
    b_ptr  = &b_ptr[bx * BN];
    c_ptr  = &c_ptr[by * BM * N + bx * BN];

    __shared__ float shared_a_ptr[BM][BK];
    __shared__ float shared_b_ptr[BK][BN];
    float            temp[NUM_PER_THREAD] = {0.f}; // 每个线程处理连续的四个数的计算，所以temp是一维的

    /*
        tx = 0 ty = 0                                  tx = 1 ty = 0
            负责 32*32 里的 a_ptr_start  [0, 3]             负责 32*32 里的 a_ptr_start [4, 7]
        tx = 0 ty = 1                                  tx = 1 ty = 1
            a_ptr_start [32, 35]                          [ty * 32 + tx * 4] [ty * 32 + tx * 4 + 4]
            [ty * 32 + tx] [ty * 32 + tx + 4]
    */
    // global ---> shared
    for (int k = 0; k < K; k += BK) {

        FETCH_FLOAT4(shared_a_ptr[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(a_ptr[ty * K + k + tx * NUM_PER_THREAD]);
        FETCH_FLOAT4(shared_b_ptr[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(b_ptr[(ty + k) * N + tx * NUM_PER_THREAD]);

        __syncthreads();

        for (int i = 0; i < NUM_PER_THREAD; i++) {
            for (int k = 0; k < BK; k++) {
                temp[i] += shared_a_ptr[ty][k] * shared_b_ptr[k][tx * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < NUM_PER_THREAD; i++) {
        c_ptr[ty * N + tx * NUM_PER_THREAD + i] = temp[i];
    }
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

    constexpr int bm             = 32;
    constexpr int bk             = 32;
    constexpr int bn             = 32;
    constexpr int NUM_PER_THREAD = 4;
    dim3          Grid(n / bn, m / bm); //
    dim3          Block(8, 32);         // Block 内部的线程数改成 8*32

#ifdef PERF_TEST_ENABLED
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif

    sgemm<bm, bn, bk, NUM_PER_THREAD><<<Grid, Block>>>(matrix_a_device, matrix_b_device, matrix_c_device, m, n, k);

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