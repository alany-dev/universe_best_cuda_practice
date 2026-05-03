#include <chrono>  // for timing
#include <cmath>   // for INFINITY
#include <cstdlib> // for malloc/free
#include <iostream>

/*
    Softmax

    给定输入向量 z = [z_1, z_2, ... , z_n] ∈ R^n

    softmax(z_i) = e^(z_i) / sum[j=1...n](e^*(z_j))，其中 i = 1,2,...,n

*/

/*
    CUDA kernel
    计算 N 个长度为 C 的向量的 softmax

    v0：
        1个线程块，N个线程并行。每个线程处理 1 行数据（行内串行遍历 C 个元素）。

        Results match: YES
        CPU time: 2.0414 ms
        GPU time: 2.2319 ms
        Speedup: 0.914646x

    v1 改动：
        - 改成 N 个线程块，每个线程块 128 个线程数（这里的线程数肯定不能无限大，必须有限制），负责对应位置的 softmax
        - 一个 线程块内，共享 shared_memory，加速计算 max 最大值 和 sum 累加和

        Results match: YES
        CPU time: 3.50997 ms
        GPU time: 1.31674 ms
        Speedup: 2.66566x

        瓶颈：
            - 全局显存 读写耗时长，大概率是最开始的 线程粗化 maxval 加载。

    v2 改动：
        - 一个线程块 会被 硬件划分为多个 线程束。
        - 每个 warp 内部包含 32 个线程，这些线程被称为 束内线程（lane）。一个 warp 中的 32 个线程并发执行相同的指令。
        - 每个 warp 中 内部，可以使用 shfl 通信原语，通过 寄存器实现数据交换。

        - 一个线程块改成 32 个线程
        - 去掉了 共享内存，

        瓶颈：
        - 并行度很差，只有 32 个线程。

        Results match: YES
        CPU time: 1.97762 ms
        GPU time: 1.02288 ms
        Speedup: 1.93338x

    v3 改动：
        - 4个warp，一共 128 个线程（和 v2 一致）
        - shared_mm 记录 4 个 warp 的 最大数 和 累加和

        Results match: YES
        CPU time: 4.40559 ms
        GPU time: 0.869216 ms
        Speedup: 5.06847x

        - 改成 1024 线程（需要考虑 实际显卡性能，可能会更好或更差）
*/
__device__ float warpReduceMax(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <unsigned int warpsPerBlock> __global__ void softmax_forward_kernel(float *out, const float *inp, int N, int C)
{
    const int tx         = threadIdx.x; // ranges [0, block_size)
    const int bx         = blockIdx.x;  // ranges [0, N)
    const int block_size = blockDim.x;  // block_size 128
    const int warpId     = tx / 32;
    const int laneId     = tx % 32;
    // const int warpsPerBlock = block_size / 32;

    out = &out[bx * C];
    inp = &inp[bx * C];
    __shared__ float shared_mm[warpsPerBlock]; // 4

    // thread coarsening 线程粗化
    float maxval = -INFINITY;
    for (int i = tx; i < C; i += block_size) {
        maxval = fmaxf(maxval, inp[i]);
    }
    maxval = warpReduceMax(maxval);

    if (laneId == 0)
        shared_mm[warpId] = maxval;
    __syncthreads();

    if (tx == 0) {
        float val = shared_mm[tx];
#pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, shared_mm[i]);
        }
        shared_mm[0] = val;
    }
    __syncthreads();

    // 每个线程 从 Warp 中的第 0 号线程那里获取数据
    float offset = shared_mm[0];

    // 同样是 一个线程负责多个数
    for (int i = tx; i < C; i += block_size) {
        out[i] = expf(inp[i] - offset);
    }

    // thread coarsening again, for the sum
    // 再次 线程粗化，计算 总和
    float sumval = 0.f;
    for (int i = tx; i < C; i += block_size) {
        sumval += out[i];
    }
    sumval = warpReduceSum(sumval);

    if (laneId == 0)
        shared_mm[warpId] = sumval;
    __syncthreads();

    if (tx == 0) {
        float val = shared_mm[tx];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += shared_mm[i];
        }
        shared_mm[0] = val;
    }
    __syncthreads();

    float sum = shared_mm[0];

    for (int i = tx; i < C; i += block_size) {
        out[i] /= sum;
    }
}

/*
    CPU
    计算 N 个 长度为C的向量的 softmax
*/
void softmax_forward_cpu(float *out, const float *inp, int N, int C)
{
    for (int i = 0; i < N; i++) {
        const float *inp_row = inp + i * C;
        float       *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }

        float sum = 0.f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        float norm = 1.f / sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}

// Function to compare results
bool compare_results(const float *cpu, const float *gpu, int N, int C, float epsilon = 1e-3f)
{
    for (int i = 0; i < N * C; ++i) {
        if (fabs(cpu[i] - gpu[i]) > epsilon) {
            std::cout << "Difference at index " << i << ": CPU=" << cpu[i] << ", GPU=" << gpu[i]
                      << ", diff=" << fabs(cpu[i] - gpu[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // Example: batch size N=32, classes C=4096
    constexpr int N = 32;
    constexpr int C = 4096;

    size_t num_elements = N * C;
    float *inp          = (float *)malloc(num_elements * sizeof(float));
    float *out_cpu      = (float *)malloc(num_elements * sizeof(float));
    float *out_gpu      = (float *)malloc(num_elements * sizeof(float));

    // Initialize input with sample data
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            inp[n * C + c] = float(c);
        }
    }

#ifdef PERF_TEST_ENABLED
    // Run CPU version and measure time
    auto start_cpu = std::chrono::high_resolution_clock::now();
#endif

    softmax_forward_cpu(out_cpu, inp, N, C);

#ifdef PERF_TEST_ENABLED
    auto                                      end_cpu  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
#endif

#ifdef PERF_TEST_ENABLED
    // Run GPU version and measure time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    float *d_out, *d_inp;
    cudaMalloc((void **)&d_out, N * C * sizeof(float));
    cudaMalloc((void **)&d_inp, N * C * sizeof(float));
    cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);

#ifdef PERF_TEST_ENABLED
    cudaEventRecord(start);
#endif

    // Launch kernel
    constexpr unsigned int blockSize     = 128;
    constexpr unsigned int warpsPerBlock = blockSize / 32;
    constexpr unsigned int numBlocks     = N;
    softmax_forward_kernel<warpsPerBlock><<<numBlocks, blockSize>>>(d_out, d_inp, N, C);

#ifdef PERF_TEST_ENABLED
    cudaEventRecord(stop);

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate milliseconds
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
#endif

    // Copy result back to host
    cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    bool success = compare_results(out_cpu, out_gpu, N, C);
    std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

#ifdef PERF_TEST_ENABLED

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print performance comparison
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x" << std::endl;
#endif

    // Cleanup
    cudaFree(d_out);
    cudaFree(d_inp);

    // Cleanup
    free(inp);
    free(out_cpu);
    free(out_gpu);

    return 0;
}