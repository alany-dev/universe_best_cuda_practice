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
*/
template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int TM, unsigned int TN>
__global__ void mysgemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const int bx               = blockIdx.x;
    const int by               = blockIdx.y;
    const int tid              = threadIdx.x;                         // [0, 256)
    const int block_row_thread = BN / TN;                             // 每行需要多少个线程
    const int block_col_thread = BM / TM;                             // 每列需要多少个线程
    int       thread_num       = block_row_thread * block_col_thread; // 计算 Block 所有数据 需要多少个线程

    // 当前线程 tid，负责 数据的起始坐标（tx,ty）到（tx+TN,ty+TM）矩形
    // 二维起始位置
    int tx = (tid % block_row_thread) * TN;
    int ty = (tid / block_row_thread) * TM;

    __shared__ float As[BM * BK]; // 128 行 8 列
    __shared__ float Bs[BK * BN]; // 8 行 128 列

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // 本质上还是 离散 的搬运数据
    /*
        128 行 8 列，一共 256 个线程
        每一列 可以分到 256 / 8 = 32 个线程
        每个线程搬该列上每隔 32 行的一个元素。

        线程0 搬行: 0, 32, 64, 96
        线程1 搬行: 1, 33, 65, 97
        线程2 搬行: 2, 34, 66, 98
        ...
        线程31搬行: 31, 63, 95, 127
    */
    int a_tile_row    = threadIdx.x / BK; // 线程在 As 中的起始行
    int a_tile_col    = threadIdx.x % BK; // 线程在 As 中的起始列
    int a_tile_stride = thread_num / BK;  // 行方向步长 = 256 / 8 = 32

    int b_tile_row    = threadIdx.x / BN; // 线程在 Bs 中的起始行
    int b_tile_col    = threadIdx.x % BN; // 线程在 Bs 中的起始列
    int b_tile_stride = thread_num / BN;  // 行方向步长 = 256 / 128 = 2

    float tmp[TM][TN] = {0.f};
#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        // 256 个线程负责 搬运 128*8 矩阵
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int j = 0; j < BK; j += b_tile_stride) {
            Bs[(b_tile_row + j) * BN + b_tile_col] = B[(b_tile_row + j) * N + b_tile_col];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int t = 0; t < BK; t++) {
#pragma unroll
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    tmp[i][j] += As[(ty + i) * BK + t] * Bs[t * BN + tx + j];
                }
            }
        }

        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++)
            C[(ty + i) * N + tx + j] = alpha * tmp[i][j] + beta * C[(ty + i) * N + tx + j];
    }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

int main()
{
    const unsigned int BLOCK_SIZE = 32;
    std::vector<int>   sizes      = {128, 256, 512, 1024, 2048, 4096, 8192};

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v2.csv");
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
