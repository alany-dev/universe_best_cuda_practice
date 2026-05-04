#include <cmath> // for fabsf
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char *msg)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*
    v1 优化：
        - 依旧是 每个线程负责 （ty,tx） 坐标数值 的 计算，每个 Block 有 32 * 32 个线程
        - 使用 BM*BK BK*BN 的共享内存 （32*32）。一个线程块内共享

    v2 优化：
        - 一个线程块 256 个线程
        - 一个线程干更多的事情
        - BM 128, BN 128, BK 8, TM 8, TN 8
            - 每个线程块处理 128 * 128 个数据。每个线程需要处理 8*8 个数据

        - global--->shared 本质上还是 离散 的搬运数据
        - 实际计算改动：内积 ---> 外积

    v3 优化：
        - global--->shared 使用 float4 加速搬运
        - A --> As 转置保存
*/

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define OFFSET(row, col, ld)  ((row) * (ld) + (col))

template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int TM, unsigned int TN>
__global__ void mysgemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const int bx               = blockIdx.x;
    const int by               = blockIdx.y;
    const int tid              = threadIdx.x;                         // [0, 256)
    const int block_row_thread = BN / TN;                             // 每行需要多少个线程
    const int block_col_thread = BM / TM;                             // 每列需要多少个线程
    const int thread_num       = block_row_thread * block_col_thread; // 计算 Block 所有数据 需要多少个线程

    // 当前线程 tid，负责 数据的起始坐标（tx,ty）到（tx+TN,ty+TM）矩形
    // 二维起始位置
    const int tx = (tid % block_row_thread) * TN;
    const int ty = (tid / block_row_thread) * TM;

    __shared__ float As[BM * BK]; // 128 行 8 列
    __shared__ float Bs[BK * BN]; // 8 行 128 列

    // (8 * 128) / 256 / 4 = 1
    const int ldg_a_num = BK * BM / thread_num / 4; // 每个线程需要搬运的 float4 的个数
    const int ldg_b_num = BK * BN / thread_num / 4;

    int a_tile_row = tid / (BK / 4);     // 线程在 As 中的起始行     0~127
    int a_tile_col = tid % (BK / 4) * 4; // 线程在 As 中的起始列     0 或 4
    int a_tile_stride =
        BM / ldg_a_num; // 行方向步长 128 / 1 = 128  😅 不好理解（其实就是 要保证 循环 ldg_a_num 次拷贝）

    int b_tile_row    = tid / (BN / 4);     // 线程在 Bs 中的起始行
    int b_tile_col    = tid % (BN / 4) * 4; // 线程在 Bs 中的起始列
    int b_tile_stride = BK / ldg_b_num;     // 行方向步长 128 / 1 = 128

    float accum[TM][TN]            = {0.f};
    float ldg_a_reg[4 * ldg_a_num] = {0.f};
    float a_frag[TM];
    float b_frag[TN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4; // 当前线程要处理 ldg_a_num 个 float4 数据，这个第几个 float4 数据。
            FETCH_FLOAT4(ldg_a_reg[ldg_index])             = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            As[OFFSET(a_tile_col, a_tile_row + i, BM)]     = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 1];
            As[OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 2];
            As[OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 3];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            // 这里不需要 寄存器缓存，因为本身就是 行排序--行处理
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }

        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int t = 0; t < BK; t++) {  
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(t, ty + m, BM)]);
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(t, tx + n, BN)]);
            }

#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);

            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;

            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

int main()
{
    const unsigned int BLOCK_SIZE = 32;
    std::vector<int>   sizes      = {128, 256, 512, 1024, 2048, 4096, 8192};

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v3.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N : sizes) {
        std::cout << "Testing size: " << N << std::endl;

        size_t size     = N * N * sizeof(float);
        float *A        = (float *)malloc(size);
        float *B        = (float *)malloc(size);
        float *C_cublas = (float *)malloc(size);
        float *C_v1     = (float *)malloc(size);

        float *d_A, *d_B, *d_C_v1;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

        bool out_of_memory = false;

        try {
            // 初始化矩阵 A 和 B
            for (int i = 0; i < N * N; ++i) {
                A[i] = 1.0f;
                B[i] = 2.0f;
            }

            // 拷贝到设备
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice), "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice), "cudaMemcpy B to device failed");

            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta  = 0.0f;

            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // warmup
            int warpup_time = 10; // 热身次数
            for (int i = 0; i < warpup_time; ++i) {
                checkCublasError(
                    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                    "cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cuBLAS SGEMM
            int repeat_time = 5;
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start cublas) failed");
            for (int i = 0; i < repeat_time; ++i) {
                checkCublasError(
                    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                    "cublasSgemm failed");
            }

            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop cublas) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize cublas failed");

            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop), "cudaEventElapsedTime cublas failed");

            // 拷贝 cuBLAS 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_cublas failed");

            // mysgemm_v1
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warpup_time; ++i) {
                mysgemm<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaDeviceSynchronize();

            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");

            for (int i = 0; i < repeat_time; ++i) {
                mysgemm<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }

            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize v1 failed");

            float v1_time = 0;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop), "cudaEventElapsedTime v1 failed");

            // 拷贝手写 kernel 结果
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost), "cudaMemcpy C_v1 failed");
            // 结果比较
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; ++i) {
                if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
                    error_count++;
                }
            }

            float cublas_gflops = repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
            float v1_gflops     = repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);     // GFlops
            // 写入CSV
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << "," << (error_count == 0 ? "1" : "0")
                     << std::endl;

            // 释放资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v1);

            free(A);
            free(B);
            free(C_cublas);
            free(C_v1);
        }
        catch (...) {
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        if (!out_of_memory) {
            std::cout << "Finished size: " << N << std::endl;
        }
        else {
            csv_file << N << ",OOM,OOM,0" << std::endl;
        }
    }

    csv_file.close();

    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'" << std::endl;
    return 0;
}
