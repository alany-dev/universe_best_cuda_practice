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

    v5 版本改动：
        - 讲 矩阵内积 改成 矩阵外积
            原本是 M N K 三次循环
            改成 K M N
        - 使用 寄存器 存储 临时的 外积矩阵。

        割裂点：
        - 在数据 global --> shared 的时候用的是 float4 所以，一个block线程块是 32*8
        - 实际进行内积，每个线程负责计算 2*2 的数据（而不是1*4），
            所以需要对 256 个线程进行重新 排序，让每个线程处理相应的 2*2。

    v6 版本改动：
        之前版本，主要是在 global --> shared 过程中 使用 float4 取数优化

        现在，我需要在 计算内积的 时候，也用到 float4
        对于 A 的 shared_a 目前还是 单独取数，在下个版本会 先转置，再float4取数。
        B 的 shared_a 可以 直接 float4 取数

        变动：
        - Block 16*16
        - num_per_block 64*64
        - 也就是每个线程需要处理连续的 4*4 个数据。

    v7 版本：
        A 从 global ---> shared 的过程中，转置赋值

    v8 版本：
        流水线
        shared_memory 的尺寸 从 64*64 改成了 128*8，每个线程 按照 流水线交替处理 float4 的数据（global->shared）


*/

// 这么写，好处是 pointer 意图接受一个左值，而不仅仅是一个指针
// 取地址 -> 强转 -> 下标访问
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// 其实 这里的 BM == blckDim.y  BN == blockDim.x    BK
// RM   每个线程 行方向 的 处理数据量   NUM_PER_THREAD 开根号 4
// RN   每个线程 列方向 的 处理数据量   NUM_PER_THREAD 开根号 4
// RK 也是 4
template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int RM, unsigned int RN>
__global__ void sgemm(float *a_ptr, float *b_ptr, float *c_ptr, const int M, const int N, const int K)
{
    int       tx               = threadIdx.x; // 0-15
    int       ty               = threadIdx.y; // 0-15
    int       bx               = blockIdx.x;
    int       by               = blockIdx.y;
    const int tid              = ty * blockDim.x + tx;
    const int a_thread_per_row = BK / 4; // 2
    const int b_thread_pre_row = BN / 4; // 32
    // block内部索引重排
    const int a_tid_x = tid % a_thread_per_row;
    const int a_tid_y = tid / a_thread_per_row;
    const int b_tid_x = tid % b_thread_pre_row;
    const int b_tid_y = tid / b_thread_pre_row;

    a_ptr = &a_ptr[by * BM * K];
    b_ptr = &b_ptr[bx * BN];
    c_ptr = &c_ptr[by * BM * N + bx * BN];

    // 改成了 128 * 8 的矩阵
    __shared__ float shared_a_ptr[2][BK][BM]; // BM * BK 改成 BK * BM，存储的是 转置后的 A
    __shared__ float shared_b_ptr[2][BK][BN];

    float a_reg[RM]      = {0.f}; // shared_a_ptr 的 竖向矩阵 RM*1
    float b_reg[RN]      = {0.f}; // shared_b_ptr 的 横向矩阵 1*RN
    float a_load_reg[RM] = {0.f}; // a 每次 float4 读取的数据，暂存
    float temp[RM][RN]   = {0.f}; // 每个线程处理 8*8，64个数

    // 流水线的装载和排空阶段不是满流水，所以不放在循环里。
    FETCH_FLOAT4(a_load_reg[0]) = FETCH_FLOAT4(a_ptr[a_tid_y * K + a_tid_x * 4]);
    // 转置
    shared_a_ptr[0][a_tid_x * 4][a_tid_y]     = a_load_reg[0];
    shared_a_ptr[0][a_tid_x * 4 + 1][a_tid_y] = a_load_reg[1];
    shared_a_ptr[0][a_tid_x * 4 + 2][a_tid_y] = a_load_reg[2];
    shared_a_ptr[0][a_tid_x * 4 + 3][a_tid_y] = a_load_reg[3];

    FETCH_FLOAT4(shared_b_ptr[0][b_tid_y][b_tid_x * 4]) = FETCH_FLOAT4(b_ptr[b_tid_y * N + b_tid_x * 4]);
    __syncthreads();

    int write_stage_idx = 1;
    for (int s = BK; s < K; s += BK) {
        FETCH_FLOAT4(a_load_reg[0]) = FETCH_FLOAT4(a_ptr[a_tid_y * K + a_tid_x * 4 + s]);
        // 转置
        shared_a_ptr[write_stage_idx][a_tid_x * 4][a_tid_y]     = a_load_reg[0];
        shared_a_ptr[write_stage_idx][a_tid_x * 4 + 1][a_tid_y] = a_load_reg[1];
        shared_a_ptr[write_stage_idx][a_tid_x * 4 + 2][a_tid_y] = a_load_reg[2];
        shared_a_ptr[write_stage_idx][a_tid_x * 4 + 3][a_tid_y] = a_load_reg[3];

        FETCH_FLOAT4(shared_b_ptr[write_stage_idx][b_tid_y][b_tid_x * 4]) =
            FETCH_FLOAT4(b_ptr[(b_tid_y + s) * N + b_tid_x * 4]);

        write_stage_idx = write_stage_idx ^ 1;

        for (int k = 0; k < BK; k++) {
            // shared_memory -> reg
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(shared_a_ptr[write_stage_idx][k][ty * RM]);
            FETCH_FLOAT4(a_reg[4]) = FETCH_FLOAT4(shared_a_ptr[write_stage_idx][k][ty * RM + 4]);

            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(shared_b_ptr[write_stage_idx][k][tx * RN]);
            FETCH_FLOAT4(b_reg[4]) = FETCH_FLOAT4(shared_b_ptr[write_stage_idx][k][tx * RN + 4]);

            for (int i = 0; i < RM; i++) {
                for (int j = 0; j < RN; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    write_stage_idx = write_stage_idx ^ 1;
    for (int k = 0; k < BK; k++) {
        FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(shared_a_ptr[write_stage_idx][k][ty * RM]);
        FETCH_FLOAT4(a_reg[4]) = FETCH_FLOAT4(shared_a_ptr[write_stage_idx][k][ty * RM + 4]);
        FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(shared_b_ptr[write_stage_idx][k][tx * RN]);
        FETCH_FLOAT4(b_reg[4]) = FETCH_FLOAT4(shared_b_ptr[write_stage_idx][k][tx * RN + 4]);
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                temp[i][j] += a_reg[i] * b_reg[j];
            }
        }
    }

    // 每个线程处理了 8*8 个数据，所以 遍历 8行，每次 转移 两个 float4
    for (int i = 0; i < RM; i++) {
        FETCH_FLOAT4(c_ptr[N * (ty * RM + i) + tx * RN])     = FETCH_FLOAT4(temp[i][0]);
        FETCH_FLOAT4(c_ptr[N * (ty * RM + i) + tx * RN + 4]) = FETCH_FLOAT4(temp[i][4]);
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

    // shared_mem 从 64 * 64 改成了 128 * 8
    // 每个线程要处理的数据 从 4*4 --> 8*8
    constexpr int bm = 128;
    constexpr int bk = 8;
    constexpr int bn = 128;
    constexpr int rm = 8;
    constexpr int rn = 8;
    dim3          Grid(n / bn, m / bm); //
    dim3          Block(16, 16);        // Block 内部的线程数

#ifdef PERF_TEST_ENABLED
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif

    sgemm<bm, bn, bk, rm, rn><<<Grid, Block>>>(matrix_a_device, matrix_b_device, matrix_c_device, m, n, k);

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