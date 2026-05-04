#include <cmath> // for fabsf
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define BLOCK_SIZE            128
#define TOL                   1e-5f
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define OFFSET(row, col, ld)  ((row) * (ld) + (col))


/*




*/

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


// ---------------------------------------------------------
// 功能：将全局内存（GMEM）中的数据加载到共享内存（SMEM）
// ---------------------------------------------------------
// 模板参数说明：
//   BM, BN, BK        : 线程块级别的分块尺寸（Block Tile），
//                        BM 对应 M 方向块大小，BN 对应 N 方向，BK 对应 K 方向（归约维度）
//   row_stride_a      : 加载矩阵 A 的 tile 时，每个线程在 M 方向上的跨步（单位：行）
//   row_stride_b      : 加载矩阵 B 的 tile 时，每个线程在 K 方向上的跨步
// 普通参数：
//   N, K             : 原始矩阵的维度 N 和 K（矩阵 A: M×K, B: K×N，但参数只用了 K 和 N）
//   A, B             : 指向全局内存中当前 tile 起始位置的指针
//   As, Bs           : 共享内存数组（A 为 BM×BK 的列主序排布，B 为 BK×BN 的行主序排布）
//   inner_row_a, inner_col_a : 当前线程负责搬运的 A tile 内的起始行/列
//   inner_row_b, inner_col_b : 当前线程负责搬运的 B tile 内的起始行/列
template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int          N,
                               int          K,
                               const float *A,
                               const float *B,
                               float       *As,
                               float       *Bs,
                               int          inner_row_a,
                               int          inner_col_a,
                               int          inner_row_b,
                               int          inner_col_b)
{
    // ---------- 加载矩阵 A 的 tile（BM×BK）到共享内存 As ----------
    // A 的全局排布：行主序，维度为 M×K。当前 tile 起始行 = inner_row_a，起始列 = inner_col_a*4
    // 每次用 float4 向量化加载 4 个元素，因此列索引每次步进 4。
    // 循环沿 M 方向（行方向）移动，步幅为 row_stride_a，直到超出 tile 高度 BM。
    for (uint off_set = 0; off_set + row_stride_a <= BM; off_set += row_stride_a) {
        // 从全局内存中一次读取 4 个连续 float（float4）
        const float4 tmp = reinterpret_cast<const float4 *>(&A[(inner_row_a + off_set) * K + inner_col_a * 4])[0];
        // 将读取的 4 个元素分散写入共享内存 As。
        // As 的排布：列主序，即同一列的元素连续存放（列优先）。这样做可以让后续 warp tile 访存时实现广播或减少 bank
        // conflict。 索引公式：(inner_col_a * 4 + element_offset) * BM + (inner_row_a + off_set)
        As[(inner_col_a * 4 + 0) * BM + inner_row_a + off_set] = tmp.x;
        As[(inner_col_a * 4 + 1) * BM + inner_row_a + off_set] = tmp.y;
        As[(inner_col_a * 4 + 2) * BM + inner_row_a + off_set] = tmp.z;
        As[(inner_col_a * 4 + 3) * BM + inner_row_a + off_set] = tmp.w;
    }

    // ---------- 加载矩阵 B 的 tile（BK×BN）到共享内存 Bs ----------
    // B 的全局排布：行主序，维度为 K×N。当前 tile 起始行 = inner_row_b，起始列 = inner_col_b*4
    // 共享内存 Bs 保持行主序排布，因此可以直接用 float4 整块拷贝，无需转置。
    // 循环沿 K 方向（行方向）移动，步幅为 row_stride_b。
    for (uint off_set = 0; off_set + row_stride_b <= BK; off_set += row_stride_b) {
        // 将全局内存中的 4 个 float 直接赋值给共享内存中对应的 4 个位置（float4 写入）
        reinterpret_cast<float4 *>(&Bs[(inner_row_b + off_set) * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4 *>(&B[(inner_row_b + off_set) * N + inner_col_b * 4])[0];
    }
}

// ---------------------------------------------------------
// 功能：从共享内存（SMEM）加载数据到寄存器，并执行乘加运算
// ---------------------------------------------------------
// 模板参数说明：
//   BM, BN, BK        : 线程块 tile 尺寸
//   WM, WN            : 单个 warp 负责计算的子 tile 尺寸（Warp Tile）
//   WMITER, WNITER    : warp tile 在 M 和 N 方向上进一步划分为更小的子块的数量（warp subtile 迭代次数）
//   WSUBM, WSUBN      : 每个 warp subtile 的尺寸（WSUBM = WM/WMITER, WSUBN = WN/WNITER）
//   TM, TN            : 每个线程在 M 和 N 方向上负责计算的元素个数（Thread Tile）
// 寄存器数组：
//   reg_m             : 暂存从 As 读取的一维片段（长度为 WMITER*TM，对应同一行 warp subtile 内线程私有的元素）
//   reg_n             : 暂存从 Bs 读取的一维片段（长度为 WNITER*TN）
//   thread_results    : 累加器数组，存放当前线程在 K 方向累积的部分和，
//                       尺寸为 (WMITER*TM) * (WNITER*TN)
// warp tile 坐标：
//   warp_row, warp_col : 当前 warp 在线程块内的行、列索引
// 线程在 warp subtile 中的局部坐标：
//   thread_row_in_warp, thread_col_in_warp
template <const int BM,
          const int BN,
          const int BK,
          const int WM,
          const int WN,
          const int WMITER,
          const int WNITER,
          const int WSUBM,
          const int WSUBN,
          const int TM,
          const int TN>
__device__ void process_from_smem(float       *reg_m,
                                  float       *reg_n,
                                  float       *thread_results,
                                  const float *As,
                                  const float *Bs,
                                  const uint   warp_row,
                                  const uint   warp_col,
                                  const uint   thread_row_in_warp,
                                  const uint   thread_col_in_warp)
{
    // 遍历归约维度 K（沿着 BK 方向的所有点积片段）
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
        // -- --加载 A 的对应子块到寄存器 reg_m-- --
        // 对 warp tile 在 M 方向上的每个 subtile 迭代
        for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
            // TM 是每个线程在 M 方向拥有的元素个数，使用分散加载
            for (uint i = 0; i < TM; ++i) {
                // As 的排布：(列索引) * BM + (行索引)
                // 当前加载位置对应：
                //   列 = dot_idx （因为 BK 维度在 As 中是列，As 列主序，列索引就是 K 方向）
                //   行 = warp_row * WM + w_sub_row_idx * WSUBM + thread_row_in_warp * TM + i
                reg_m[w_sub_row_idx * TM + i] =
                    As[(dot_idx * BM) + warp_row * WM + w_sub_row_idx * WSUBM + thread_row_in_warp * TM + i];
            }
        }

        // ---- 加载 B 的对应子块到寄存器 reg_n ----
        // Bs 排布：行主序，大小为 BK × BN。
        for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
            for (uint i = 0; i < TN; ++i) {
                // Bs 行主序，索引公式为 Bs[row * BN + col]
                // row = dot_idx （K 方向）
                // col = warp_col * WN + w_sub_col_idx * WSUBN + thread_col_in_warp * TN + i
                reg_n[w_sub_col_idx * TN + i] =
                    Bs[(dot_idx * BN) + warp_col * WN + w_sub_col_idx * WSUBN + thread_col_in_warp * TN + i];
            }
        }

        // ---- 执行外积计算，累加到 thread_results ----
        for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
            for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                        // 累加器索引展平：将二维 (sub_row, sub_col, TM, TN) 映射到一维
                        thread_results[(w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) + (w_sub_col_idx * TN)
                                       + res_idx_n] +=
                            reg_m[w_sub_row_idx * TM + res_idx_m] * reg_n[w_sub_col_idx * TN + res_idx_n];
                    }
                }
            }
        }
    }
}

// 单 warp 线程数
constexpr int WARP_SIZE = 32;

// ---------------------------------------------------------
// 核心 Kernel：基于 warp tiling 的 SGEMM
// ---------------------------------------------------------
// 模板参数：
//   BM, BN, BK        : 线程块 tile 尺寸（Block Tile）
//   WM, WN            : warp tile 尺寸
//   WNITER            : warp tile 在 N 方向的 subtile 数量（M 方向由编译器自动推导）
//   TM, TN            : 每个线程负责的 M、N 方向元素个数
//   NUM_THREADS       : 线程块内的线程总数
// launch_bounds 指定每个线程块期望的线程数，帮助编译器优化寄存器使用。
template <const int BM,
          const int BN,
          const int BK,
          const int WM,
          const int WN,
          const int WNITER,
          const int TM,
          const int TN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    mysgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const uint by = blockIdx.y;
    const uint bx = blockIdx.x;

    // 线程在线程块内的全局索引（0 ~ NUM_THREADS-1）
    const uint warp_idx = threadIdx.x / WARP_SIZE; // 当前线程所在的 warp 编号
    // 根据 warp 编号计算该 warp 负责的 warp tile 位置（列优先排布）
    // 线程块内有 (BN/WN) 列 warp 和 (BM/WM) 行 warp
    const uint warp_col = warp_idx % (BN / WN); // warp 在 N 方向上的索引，行
    const uint warp_row = warp_idx / (BN / WN); // warp 在 M 方向上的索引，列

    // WNITER = 4 意味着 一个 Warp 要处理的 WM * WN （64 * 64）需要划分成 4 个 （64 * 16）的子块
    // 这里能保证，每个 warp 的 多个子块内部的 可以被 32 个线程（每个处理TM*TN数据）全部处理完毕
    constexpr uint WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
    constexpr uint WSUBM  = WM / WMITER; // 每个 M 方向子块的 高度  64
    constexpr uint WSUBN  = WN / WNITER; // 每个 N 方向子块的 宽度  16
    // 换个角度，每个线程要处理 WMITER * WNITER 次的 TM * TN 的 数据。

    // 获取线程在 warp 内部的局部 ID (0~31)
    const uint thread_idx_in_warp = threadIdx.x % WARP_SIZE;
    // 线程在 warp subtile 内负责的列索引（N 方向）
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    // 线程在 warp subtile 内负责的行索引（M 方向）
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);
    // 这里就不需要 stride 了，32 个线程，每个线程处理完 TM * TN (8 * 4) 就可以处理玩一个 子块。

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += by * BM * K;
    B += bx * BN;
    // C 的维度为 M×N，起始地址跳到当前 warp 负责的子块（之前是以 Block 分块作为起始）：
    //   c_row * BM 是该 block 对应的 C 行起始，加上 warp_row * WM 是 warp 内部偏移
    //   c_col * BN 是 block 列起始，加上 warp_col * WN 是 warp 内部偏移
    C += (by * BM + warp_row * WM) * N + bx * BN + warp_col * WN;

    // Global -> Shared 的转移操作和之前一致
    // ----- 计算每个线程在加载全局内存到共享内存时的角色 -----
    // 对于 A tile (BM×BK)：将线程映射到 (M 方向行, K 方向列/4) 二维空间
    // inner_row_a = threadIdx.x / (BK/4)   -> M 方向的行索引
    // inner_col_a = threadIdx.x % (BK/4)   -> K 方向的“列块”索引（每块含 4 个元素）
    // row_stride_a = (NUM_THREADS * 4) / BK
    //              -> 理解： NUM_THREAD * 4 表示，整个线程块（128线程）一次共同的协作读取float数量。
    //   (NUM_THREADS * 4) / BK  除以 BK 表示，所有线程并发读一次，恰好能填满 A 分块的多少行（也就可以得到每次的偏移）
    const uint inner_row_a    = threadIdx.x / (BK / 4);
    const uint inner_col_a    = threadIdx.x % (BK / 4);
    const uint row_stride_a = (NUM_THREADS * 4) / BK;

    const uint inner_row_b    = threadIdx.x / (BN / 4);
    const uint inner_col_b    = threadIdx.x % (BN / 4);
    const uint row_stride_b = (NUM_THREADS * 4) / BN;

    // 每个线程 寄存器数组
    // 在整个 Warp 32 个线程里，要处理 WM * WN 个数据（一次性处理不完）。分成了 WNITER * WMITER 子块
    // 内部的序号编排其实可以看作是一个 WMITER*TM 行 WNITER * TN 列的数组 ⭐
    float thread_results[WMITER * TM * WNITER * TN] = {0.f};

    float reg_m[WMITER * TM] = {0.0};
    float reg_n[WNITER * TN] = {0.0};

    // ----- 主循环：沿 K 维度滑动，每次处理 BK 个元素 -----
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK) {
        // 1. 协作加载当前 block tile 所需的 A 和 B 的子矩阵到共享内存
        load_from_gmem<BM, BN, BK, row_stride_a, row_stride_b>(
            N, K, A, B, As, Bs, inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        // 2. 同步，确保共享内存数据全部写完
        __syncthreads();

        // 3. 各个 warp 从共享内存读取自己的数据，执行乘加
        process_from_smem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            reg_m, reg_n, thread_results, As, Bs, warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        // 4. 移动 A、B 全局指针到下一个 BK 块
        A += BK;         // A 沿 K 方向移动
        B += BK * N;     // B 沿 K 方向移动（B 是行主序，每行长度为 N）
        __syncthreads(); // 下一次加载前需要同步，避免覆盖正在使用的共享内存
    }

    // 写回 C
    // 按照 WMITER x WNITER 写回，主要是 按照 线程的处理数据写回 这很好理解
    for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
        for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
            // 该 subtile 对应的 C 相对地址（ C地址已经偏移到了 当前 warp 负责的子块）
            float *C_interim = C + OFFSET(w_sub_row_idx * WSUBM, w_sub_col_idx * WSUBN, N);
            // 遍历该线程负责的 M 方向元素 (TM 个) 和 N 方向元素 (TN 个，每次处理 4 个)
            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1) {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) {
                    float4 tmp = FETCH_FLOAT4(
                        C_interim[OFFSET(thread_row_in_warp * TM + res_idx_m, thread_col_in_warp * TN + res_idx_n, N)]);
                    // 计算 thread_results 数组中的线性索引
                    const int i = OFFSET(w_sub_row_idx * TM + res_idx_m, w_sub_col_idx * TN + res_idx_n, WNITER * TN);
                    // alpha*accumulator + beta*original
                    tmp.x = alpha * thread_results[i + 0] + beta * tmp.x;
                    tmp.y = alpha * thread_results[i + 1] + beta * tmp.y;
                    tmp.z = alpha * thread_results[i + 2] + beta * tmp.z;
                    tmp.w = alpha * thread_results[i + 3] + beta * tmp.w;
                    // 写回全局内存
                    FETCH_FLOAT4(C_interim[OFFSET(
                        thread_row_in_warp * TM + res_idx_m, thread_col_in_warp * TN + res_idx_n, N)]) = tmp;
                }
            }
        }
    }
}

std::vector<int> generateSizes()
{
    std::vector<int> sizes;
    for (int i = 256; i <= 8192; i += 256) {
        sizes.push_back(i);
    }
    return sizes;
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main()
{
    std::vector<int> sizes = generateSizes();

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v7.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

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

            const uint K10_NUM_THREADS = 128;
            const uint K10_BN          = 128;
            const uint K10_BM          = 128;
            const uint K10_BK          = 16;
            const uint K10_WN          = 64;
            const uint K10_WM          = 64;
            const uint K10_WNITER      = 4;
            const uint K10_TN          = 4;
            const uint K10_TM          = 8;
            dim3       blockDim(K10_NUM_THREADS);

            constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

            // warptile in threadblocktile
            static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
            static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);
            // threads in warpsubtile
            static_assert((K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) == 0);
            constexpr uint K10_WMITER = (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
            // warpsubtile in warptile
            static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

            static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                          "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                          "issues during GMEM->SMEM tiling (loading only parts of the "
                          "final row of Bs during each iteraion)");
            static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                          "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                          "issues during GMEM->SMEM tiling (loading only parts of the "
                          "final row of As during each iteration)");
            static_assert(K10_BN % (16 * K10_TN) == 0, "BN must be a multiple of 16*TN to avoid quantization effects");
            static_assert(K10_BM % (16 * K10_TM) == 0, "BM must be a multiple of 16*TM to avoid quantization effects");
            static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                          "BM*BK must be a multiple of 4*256 to vectorize loads");
            static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                          "BN*BK must be a multiple of 4*256 to vectorize loads");

            dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(N, K10_BM));

            for (int i = 0; i < warpup_time; ++i) {
                mysgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaDeviceSynchronize();

            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");
            for (int i = 0; i < repeat_time; ++i) {
                mysgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM, K10_TN, K10_NUM_THREADS>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize v1 failed");
            checkCudaError(cudaGetLastError(), "cuda get last error failed");
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

            float ratio = v1_gflops / cublas_gflops;
            // 写入CSV
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << "," << (error_count == 0 ? "1" : "0") << ","
                     << ratio << std::endl;

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
            cudaDeviceSynchronize();
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
