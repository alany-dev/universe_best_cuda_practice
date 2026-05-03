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
*/

/*
1. 分层并行：块间处理样本，块内处理元素
每个线程块负责一个样本（一个长度为 C 的向量），总共启动 N 个块，实现样本间并行。

线程块内的多个线程（如 128 个）通过循环步幅 (i += blockDim.x) 分工合作，处理样本内所有的 C 个元素，这就是线程粗化 (thread coarsening)，既适应了 C 可能远大于线程数的情况，又减少了线程创建开销。

2. 共享内存 + 树形规约，避免全局同步
用 __shared__ 数组在线程块内协作完成求最大值和求和两步规约操作。

规约采用二分树形合并（stride 从 blockSize/2 递减到 1），每一步用 __syncthreads() 保证线程间数据同步，最终所有线程都能拿到全局最大值（或总和）。

这样两步计算（max、sum）完全在快速的共享内存中完成，无需访问全局内存，大幅降低了延迟。

3. 数值稳定的 Softmax 算法
先求每个样本所有元素的最大值，用原数据减去该最大值再计算指数：exp(x - max)，避免指数溢出（传统做法）。

将指数化后的值累加得到和，再逐元素除以该和，输出概率分布。整体流程在内核中一次完成，无需额外启动多个内核或做主机端同步。
*/
template <unsigned int BLOCK_SIZE> __global__ void softmax_forward_kernel(float *out, const float *inp, int N, int C)
{
    const int tx         = threadIdx.x; // ranges [0, block_size)
    const int bx         = blockIdx.x;  // ranges [0, N)
    const int block_size = blockDim.x;  // block_size 128

    out = &out[bx * C];
    inp = &inp[bx * C];

    __shared__ float shared_mm[BLOCK_SIZE];

    // thread coarsening 线程粗化，128个线程要处理 C个数据。
    // 比如，0 号线程 要处理 0,128,256 ... 数据的最大值，先 线程内部计算。
    float maxval = -INFINITY;
    for (int i = tx; i < C; i += block_size) {
        maxval = fmaxf(maxval, inp[i]);
    }

    shared_mm[tx] = maxval; // shared_memory 记录了 128 个线程的 局部最大值，之后只需要处理这 128 个数据的最大值
    __syncthreads();

    // reductions 规约，stride 从大到小，避免了 bank conflict
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tx < stride) {
            shared_mm[tx] = fmaxf(shared_mm[tx], shared_mm[tx + stride]);
        }
    }
    __syncthreads();
    // 每个线程都记录一下 最大值
    float offset = shared_mm[0];

    // 同样是 一个线程负责多个数
    for (int i = tx; i < C; i += block_size) {
        out[i] = expf(inp[i] - offset);
    }
    __syncthreads();

    // thread coarsening again, for the sum
    // 再次 线程粗化，计算 总和
    float sumval = 0.f;
    for (int i = tx; i < C; i += block_size) {
        sumval += out[i];
    }
    shared_mm[tx] = sumval; // 128 个累加和
    __syncthreads();

    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tx < stride) {
            shared_mm[tx] += shared_mm[tx + stride];
        }
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
    constexpr unsigned int blockSize = 128;
    constexpr unsigned int numBlocks = N;
    softmax_forward_kernel<blockSize><<<numBlocks, blockSize>>>(d_out, d_inp, N, C);

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