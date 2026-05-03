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
*/
__global__ void softmax_forward_kernel(float *out, const float *inp, int N, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
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
    int N = 32;
    int C = 4096;

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
    int blockSize = N;
    int numBlocks = 1;
    softmax_forward_kernel<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);

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