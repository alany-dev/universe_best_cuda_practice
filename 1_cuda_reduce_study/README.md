# 1_cuda_reduce_study 优化报告

本文记录 `1_cuda_reduce_study` 目录中 CUDA reduction 示例的优化路线。主线文件从 `my_reduce_v0_global_memory.cu` 迭代到 `my_reduce_v8_shuffle.cu`，目标是把一段 `float` 数组按 block 做局部求和，输出每个 block 的 partial sum。`reduce.cu` 是额外的两阶段全数组归约示例，和 v0-v8 的输入规模、输出语义不同，本文单独说明。

## 结论摘要

Reduction 的核心瓶颈通常不是浮点加法，而是内存访问、同步、线程利用率和中间 partial sum 数量。当前代码的优化方向是逐步把工作从慢的 global memory 转移到 shared memory 和 register，再减少无效线程、`__syncthreads()` 次数、shared memory bank conflict，最后使用 warp shuffle 替代 warp 内 shared memory 归约。

本机 Release/SM86 复核实测中，v0 到 v8 在 128 MiB 输入上的单 kernel 时间从约 `0.902 ms` 降到约 `0.277 ms`，有效带宽从约 `138.51 GB/s` 提升到约 `451.17 GB/s`。v5/v6/v8 的差距已经接近单次测量波动，不应只按 0.01 ms 级别排序；更重要的是理解各版本减少了哪类开销。

最大的性能跃迁来自三步：v3 通过 sequential addressing 避免 bank conflict；v4 在 load 阶段让每个线程先合并 2 个元素，显著减少 block 数或每 block 线程数；v5/v8 把最后一个 warp 的归约从 shared memory + barrier 改成 warp 级展开或 shuffle。

## 测试环境与口径

测试时间：2026-04-25。

测试 GPU：`NVIDIA GeForce RTX 3090`，显存 `24576 MiB`，Driver `580.95.05`。机器上有 4 张 RTX 3090，下面数据来自单进程默认 GPU。

编译口径：使用单独 Release 构建，避免当前 `build` 目录中的 Debug `-G` 对性能造成严重干扰。

```bash
cmake -S . -B /tmp/cuda_reduce_perf_build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/cuda_reduce_perf_build -j 8
```

输入规模：v0-v8 均使用 `N = 32 * 1024 * 1024` 个 `float`，即 `128 MiB` 输入。代码中的有效带宽计算为 `N * sizeof(float) / kernel_time`，没有把输出 partial sum 的写回显式计入分子；由于输出相对输入较小，这个口径适合看趋势，但不是严格的 DRAM 总流量。

正确性口径：v0-v8 都只验证每个 block 的 partial sum 是否与 CPU 逐 block 求和一致，输出不是整个数组的单个总和。`reduce.cu` 才演示两阶段全数组求和。

运行注意：如果在受限 sandbox 中运行 CUDA 程序且没有 `/dev/nvidia*` 设备节点，程序可能打印 `0.000 ms`、`inf GB/s` 或 `wrong`。这是运行环境无法访问 GPU，而不是这些 kernel 的真实性能。当前源码没有检查 CUDA API 返回值，因此这类错误会被吞掉。

## 实测性能

以下为 Release/SM86、外部 GPU 权限下的一轮复核数据，所有 v0-v8 均输出 `right`。

| 版本 | 文件 | 核函数耗时 | 有效带宽 | 相对 v0 加速 |
|---|---|---:|---:|---:|
| v0 | `my_reduce_v0_global_memory.cu` | `0.902 ms` | `138.51 GB/s` | `1.00x` |
| v1 | `my_reduce_v1_shared_memory.cu` | `0.850 ms` | `147.00 GB/s` | `1.06x` |
| v2 | `my_reduce_v2_no_divergence_branch.cu` | `0.786 ms` | `159.09 GB/s` | `1.15x` |
| v3 | `my_reduce_v3_no_bank_conflict.cu` | `0.568 ms` | `220.14 GB/s` | `1.59x` |
| v4 Plan A | `my_reduce_v4_add_during_load_plan_a.cu` | `0.347 ms` | `360.45 GB/s` | `2.60x` |
| v4 Plan B | `my_reduce_v4_add_during_load_plan_b.cu` | `0.341 ms` | `366.65 GB/s` | `2.65x` |
| v5 | `my_reduce_v5_unroll_last_warp.cu` | `0.269 ms` | `465.25 GB/s` | `3.35x` |
| v6 | `my_reduce_v6_completely_unroll.cu` | `0.268 ms` | `466.75 GB/s` | `3.37x` |
| v7 | `my_reduce_v7_mutli_add.cu` | `0.283 ms` | `441.33 GB/s` | `3.19x` |
| v8 | `my_reduce_v8_shuffle.cu` | `0.277 ms` | `451.17 GB/s` | `3.26x` |

补充参考：`reduce.cu` 使用 `N = 1024 * 1024`，两阶段 GPU 归约耗时约 `0.091584 ms`，CPU 顺序求和约 `2.36848 ms`，结果校验通过。由于输入规模和算法结构不同，它不应直接和 v0-v8 的表格横向比较。

## 背景知识

### Reduction 是典型 memory-bound 算子

对 `float` 求和时，每读入 4 字节只做一次加法，算术强度很低。GPU 的浮点算力通常远高于全局内存供数能力，因此 naive reduction 的瓶颈往往是 global memory 访问、同步和线程调度，而不是加法本身。

优化目标通常包括：让 global memory 访问合并成 coalesced transaction；减少 global memory 中间写回；把中间值放入 shared memory 或 register；减少 `__syncthreads()`；减少 warp divergence；减少 shared memory bank conflict；让每个线程做适量连续工作以提高 ILP，但不要把并行度压得太低。

### CUDA 执行层级

Grid 由多个 block 组成，block 在 SM 上调度执行。一个 block 内的线程可以用 `__syncthreads()` 同步，也可以共享 `__shared__` 内存。不同 block 之间不能在单个 kernel 内直接同步，因此 v0-v8 的设计都是每个 block 产出一个 partial sum。

Warp 是 CUDA 调度的基本执行单位，当前主流 NVIDIA GPU 的 warp 大小是 32 个线程。一个 warp 内线程执行同一条指令流，但如果条件分支让部分 lane 走不同路径，就会产生 warp divergence，硬件需要分路径串行执行，吞吐下降。

### 内存层级

Global memory 容量大但延迟高，访问需要尽量 coalesced。对 `float` 来说，一个 warp 的 32 个连续 lane 如果读取连续且对齐的 32 个 `float`，通常对应 128B 的连续访问段，硬件可以高效合并。

Shared memory 位于 SM 内，延迟远低于 global memory，适合保存 block 内中间结果。但 shared memory 被划分为 32 个 bank。对 4 字节 `float`，通常可理解为 `bank = index % 32`。同一 warp 的一次 shared memory 指令中，如果多个 lane 访问同一个 bank 的不同地址，就会发生 bank conflict，访问被拆分，吞吐下降。

Register 是每个线程私有的最快存储。v7/v8 先让每个线程在 register 中累加多个元素，再进入 block 内归约，就是在提高每个线程的局部工作量，减少 shared memory 和 block 数量。

### 同步与 warp 级归约

`__syncthreads()` 是 block 级 barrier，能保证 block 内所有线程都到达该点，并让之前的内存写入对 block 内线程可见。它正确但不免费，过多 barrier 会限制性能。

当归约规模缩小到一个 warp 内时，可以利用 warp 内线程共同执行的特性，避免 block 级同步。旧写法常用 `volatile shared memory` 展开最后 32 或 64 个元素；更现代、更清晰的写法是使用 `__shfl_down_sync()` 在 register 之间直接交换数据。

`__shfl_down_sync(mask, val, offset)` 只在同一个 warp 内工作。`mask` 表示参与本次 shuffle 的 lane 集合，参与 lane 必须都执行到这条指令。当前 v8 使用 `0xffffffff`，因为 blockDim 是 256 且每个 warp 都满 32 lane；如果处理尾部不满 warp 的数据，必须构造正确 mask 或先把无效 lane 置零并确保参与语义正确。

## 迭代详解

### v0：global memory 原地归约

文件：`my_reduce_v0_global_memory.cu`。

配置：`THREAD_PER_BLOCK = 256`，`block_num = N / 256 = 131072`。每个 block 处理 256 个输入元素，并输出 1 个 partial sum。

核心代码模式：

```cpp
for (int i = 1; i < blockDim.x; i *= 2) {
    if (tid % (i * 2) == 0) {
        input_begin[tid] += input_begin[tid + i];
    }
    __syncthreads();
}
```

优点：逻辑最直接，便于理解 reduction tree。每一轮把相邻间隔为 `i` 的元素合并，经过 `log2(blockDim.x)` 轮后由 `tid == 0` 写出 block 结果。

主要问题：中间结果直接写 global memory，延迟高且会产生大量中间写；`tid % (i * 2)` 有取模开销；active 线程呈稀疏分布，几乎每个 warp 内都混合 active/inactive lane，warp divergence 明显；每轮都要 `__syncthreads()`。

性能定位：`0.902 ms`，`138.51 GB/s`。这是后续优化的 baseline。

### v1：把中间结果放入 shared memory

文件：`my_reduce_v1_shared_memory.cu`。

变化：每个线程先从 global memory 读取一个元素到 `shared[tid]`，后续 reduction 都在 shared memory 中完成，最后只写一次 global output。

```cpp
shared[tid] = input_begin[tid];
__syncthreads();
```

解决的问题：避免每一轮都对 global memory 做读写，中间结果停留在 SM 内的 shared memory，显著降低访问延迟和全局内存压力。

仍然存在的问题：归约结构仍然使用 `tid % (i * 2) == 0`，所以 warp divergence 和取模开销还在；每轮仍然有 `__syncthreads()`。

性能变化：`0.850 ms`，相对 v0 约 `1.06x`。提升不大，说明 v0 的瓶颈不只是 global memory，中间的分支形态、同步和线程利用率也很关键。

### v2：减少早期 warp divergence

文件：`my_reduce_v2_no_divergence_branch.cu`。

变化：使用连续的低编号线程参与每轮归约，用 `tid < blockDim.x / (2 * i)` 判断 active 线程，再计算实际访问下标。

```cpp
if (tid < blockDim.x / (2 * i)) {
    int index = tid * 2 * i;
    shared[index] += shared[index + i];
}
```

优化点：active 线程从 `tid = 0` 开始连续排列。早期迭代中，一个 warp 往往是整 warp active 或整 warp inactive，比 v1 的“每个 warp 内稀疏 active”更友好。

新问题：`index = tid * 2 * i` 让同一 warp 访问 shared memory 时出现 stride 访问。以 32 bank shared memory 来看，stride 为 2、4、8 等时，多个 lane 会映射到相同 bank 的不同地址，产生 bank conflict。v2 是典型“减少 divergence 但引入 bank conflict”的中间版本。

性能变化：`0.786 ms`，相对 v0 约 `1.15x`。比 v1 更好，但 bank conflict 限制了继续提升。

### v3：sequential addressing，避免 bank conflict

文件：`my_reduce_v3_no_bank_conflict.cu`。

变化：从 `blockDim.x / 2` 开始向下折半，连续线程访问连续 shared memory 地址。

```cpp
for (int i = blockDim.x / 2; i >= 1; i /= 2) {
    if (tid < i) {
        shared[tid] += shared[tid + i];
    }
    __syncthreads();
}
```

优化点：每轮 active 线程访问 `shared[tid]` 和 `shared[tid + i]`。对一个 warp 来说，`tid` 是连续的，`tid + i` 也是连续的，因此每条 shared memory load/store 指令基本按连续 bank 分布，显著降低 bank conflict。

额外收益：不再使用取模，索引计算更简单。active 线程仍然是连续低编号线程，早期 warp divergence 也较少。

仍然存在的问题：即使归约进入最后一个 warp，代码仍然每轮使用 `__syncthreads()`，而 block 级 barrier 对 warp 内归约来说过重。

性能变化：`0.568 ms`，相对 v0 约 `1.59x`。这是第一处大幅提升，说明 shared memory bank conflict 对该版本影响很明显。

### v4 Plan A：load 阶段每线程合并 2 个元素，减少 block 数

文件：`my_reduce_v4_add_during_load_plan_a.cu`。

配置：`THREAD_PER_BLOCK = 256`，每个线程读取 2 个元素，因此每个 block 处理 `512` 个元素，`block_num = N / 256 / 2 = 65536`。

核心变化：

```cpp
float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
shared[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
```

优化点：在从 global memory 搬运到 shared memory 时就先做一次加法，把 512 个输入压缩成 256 个 shared memory 中间值。后续 block 内归约规模仍是 256，但 block 数直接减半。

内存访问特点：每个 warp 读取两段连续的 32 个 `float`。这类访问容易 coalesced，且每个线程多做一次加法可以提高指令级并行，帮助隐藏 global memory 延迟。

代价：每个 block 仍有 256 个线程，单 block 资源使用和 v3 相近。由于输出 partial sum 数量减少，若后续要做第二阶段全局归约，Plan A 会比保持 block 数的方案更有利。

性能变化：`0.347 ms`，相对 v0 约 `2.60x`。这一步收益很大，原因是减少了 block 数、同步次数总量和输出 partial sum 数量。

### v4 Plan B：保持 block 数，减少每 block 线程数

文件：`my_reduce_v4_add_during_load_plan_b.cu`。

配置：`THREAD_PER_BLOCK = 128`，每个线程读取 2 个元素，因此每个 block 仍处理 `256` 个输入元素，`block_num = N / 128 / 2 = 131072`。

优化点：保持和 v0-v3 相同的 block 数，但每个 block 从 256 线程降到 128 线程。每个 block 内参与 reduction 的线程更少，shared memory 更小，block 内同步和调度负担下降。

和 Plan A 的差异：Plan A 减少 block 数，Plan B 减少每 block 线程数。Plan B 的单 kernel 时间略快于 Plan A，但它输出的 partial sum 数量更多。如果最终目标是全数组单值归约，Plan A 后续第二阶段开销更小；如果只需要固定 256 元素粒度的 partial sum，Plan B 更合适。

性能变化：`0.341 ms`，相对 v0 约 `2.65x`。和 Plan A 接近，说明“每线程先合并 2 个元素”是关键收益来源。

### v5：展开最后一个 warp，减少 barrier

文件：`my_reduce_v5_unroll_last_warp.cu`。

变化：block 级循环只执行到 `i > 32`，最后 64 个 shared memory 值由 `tid < 32` 的一个 warp 完成展开归约。

```cpp
for (int i = blockDim.x / 2; i > 32; i /= 2) {
    if (tid < i) {
        shared[tid] += shared[tid + i];
    }
    __syncthreads();
}

if (tid < 32) {
    warpReduce(shared, tid);
}
```

`warpReduce` 手工展开：

```cpp
cache[tid] += cache[tid + 32];
cache[tid] += cache[tid + 16];
cache[tid] += cache[tid + 8];
cache[tid] += cache[tid + 4];
cache[tid] += cache[tid + 2];
cache[tid] += cache[tid + 1];
```

优化点：当归约范围进入一个 warp 后，不再需要 block 级 `__syncthreads()`。手工展开也减少了循环控制指令。

正确性注意：这里使用 `volatile float *cache` 是旧式 warp-synchronous shared memory 写法，用于避免编译器把 shared memory 访问缓存到寄存器或重排。对 Volta 之后具备 independent thread scheduling 的架构，更推荐使用 `__syncwarp()` 或直接使用 v8 的 shuffle 写法。

性能变化：`0.269 ms`，相对 v0 约 `3.35x`。这说明最后几轮频繁 barrier 和 shared memory 操作在 v4 之后已经成为重要瓶颈。

### v6：完全展开 block 级 reduction

文件：`my_reduce_v6_completely_unroll.cu`。

变化：把 v5 中 `i = 128, 64` 等循环写成编译期常量判断。

```cpp
if (THREAD_PER_BLOCK >= 256) {
    if (tid < 128) {
        shared[tid] += shared[tid + 128];
    }
    __syncthreads();
}

if (THREAD_PER_BLOCK >= 128) {
    if (tid < 64) {
        shared[tid] += shared[tid + 64];
    }
    __syncthreads();
}
```

优化点：`THREAD_PER_BLOCK` 是宏常量，编译器可以在编译期保留或删除对应分支。这样减少循环变量、比较、除法或移位等控制开销，也便于编译器调度指令。

代价：代码可读性和可扩展性下降。每支持一种 block size，都需要确保展开路径正确。当前代码固定 256 线程，因此收益明确；如果未来需要动态 block size，应改成模板参数而不是运行时变量。

性能变化：`0.268 ms`，相对 v0 约 `3.37x`。与 v5 基本相同，说明 v5 已经消除了主要开销，完全展开只带来很小边际收益。

### v7：固定 1024 个 block，每线程累加多个元素

文件：`my_reduce_v7_mutli_add.cu`。

配置：`block_num = 1024`，`THREAD_PER_BLOCK = 256`，`num_per_block = N / 1024 = 32768`，`num_per_thread = 32768 / 256 = 128`。

核心变化：

```cpp
float *input_begin = d_input + NUM_PER_BLOCK * blockIdx.x;
shared[tid] = 0;
for (int i = 0; i < NUM_PER_THREAD; i++) {
    shared[tid] += input_begin[tid + i * THREAD_PER_BLOCK];
}
```

优化点：每个线程先在本地累加 128 个元素，把大量 global memory 输入压缩成每 block 256 个 shared memory 值。block 数从 v4 Plan A 的 65536 进一步减少到 1024，输出 partial sum 也只有 1024 个。

内存访问特点：循环中固定 `i` 时，一个 warp 的 32 个 lane 读取连续 32 个 `float`，仍然是 coalesced 访问。跨 `i` 迭代时，每个线程以 `THREAD_PER_BLOCK` 为 stride 前进。

收益和代价：这种写法非常适合“还需要第二阶段归约”的场景，因为 partial sum 数量很少；但第一阶段单 kernel 不一定比 v5/v6 更快，因为每个线程的串行累加循环变长，单个 block 工作量增大，并行度下降。

性能变化：`0.283 ms`，相对 v0 约 `3.19x`。略慢于 v5/v6，但输出规模只有 1024，对完整归约流水线仍有价值。

### v8：使用 warp shuffle 做 block reduce

文件：`my_reduce_v8_shuffle.cu`。

变化：每个线程先在 register 变量 `sum` 中累加多个输入，不再把每线程的初始 partial sum 写入 shared memory。warp 内归约使用 `__shfl_down_sync()`，每个 warp 的 lane 0 把 warp sum 写入 shared memory，最后由第一个 warp 再做一次 shuffle 归约。

第一层 warp 归约：

```cpp
sum += __shfl_down_sync(0xffffffff, sum, 16);
sum += __shfl_down_sync(0xffffffff, sum, 8);
sum += __shfl_down_sync(0xffffffff, sum, 4);
sum += __shfl_down_sync(0xffffffff, sum, 2);
sum += __shfl_down_sync(0xffffffff, sum, 1);
```

跨 warp 合并：

```cpp
__shared__ float warpLevelSums[32];
if (laneId == 0) {
    warpLevelSums[warpId] = sum;
}
__syncthreads();

if (warpId == 0) {
    sum = (laneId < blockDim.x / 32) ? warpLevelSums[laneId] : 0.f;
    // 再做一次 warp shuffle reduction
}
```

优化点：warp 内数据交换发生在 register 之间，不需要 shared memory 读写，也不需要每一步 `__syncthreads()`。shared memory 只用于保存每个 warp 的一个 sum，当前 256 线程 block 只需要 8 个有效 slot。

正确性条件：`__shfl_down_sync` 只能在同一 warp 内交换。当前 blockDim 是 32 的倍数，因此每个 warp 满 lane；最终 warp0 合并 8 个 warp sum 时，其余 lane 被置零后参与 shuffle。若未来 blockDim 不是 32 的倍数或 N 不是整除关系，需要补充边界判断和 mask。

性能变化：`0.277 ms`，相对 v0 约 `3.26x`。单次略慢于 v6，但代码方向更现代，避免了 `volatile shared memory` 的隐含假设，扩展到更通用 block reduce 时更推荐。

## `reduce.cu` 的意义

`reduce.cu` 和 v0-v8 不同，它实现了更接近实际使用的两阶段全数组 reduction。

第一阶段：每个 block 使用 grid-stride loop 累加多个输入元素。

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    sum += in[i];
}
```

第二阶段：对第一阶段输出的 `num_blocks` 个 partial sum 再启动一个 block 做归约。

```cpp
reduce_v4<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
reduce_v4<<<1, num_blocks>>>(d_result, d_final_result, num_blocks);
```

这个文件的 `block_reduce` 已经采用 shuffle + shared memory 的标准结构：先每个 warp 内 shuffle 归约，再把每个 warp 的结果写到 shared memory，最后由 warp0 做第二次 shuffle 归约。它比 v0-v8 更接近可复用写法，因为它通过 grid-stride loop 支持任意 `n`，并最终输出一个标量结果。

但当前 `reduce.cu` 也有约束：第二阶段直接用 `<<<1, num_blocks>>>`，要求 `num_blocks <= 1024` 才能作为一个 CUDA block 的线程数；当前 `N = 1024 * 1024`、`BLOCK_SIZE = 1024` 正好满足。如果 N 更大，需要多轮归约或固定 block 数的 grid-stride 方案。

## 代码当前限制

v0-v8 假设 `N` 可以被当前每 block 处理元素数整除，没有处理尾部元素。真实工程代码应添加边界判断，例如 `if (global_index < N)`。

源码没有检查 CUDA API 返回值。建议增加 `CHECK_CUDA(cudaMalloc(...))`、kernel launch 后 `cudaGetLastError()`、计时前后 `cudaEvent` 返回值检查。否则设备不可用、架构不匹配、非法访问都会表现为错误结果或 0 ms。

benchmark 没有 warmup、没有多轮统计，也没有固定 GPU clock。当前数据适合教学和比较趋势，不适合作为严肃性能基准。更严谨的口径应至少包括 warmup、重复运行、均值/中位数/p95、`cudaDeviceSynchronize()`、Nsight Compute 指标，以及与 CUB `DeviceReduce` 的对比。

v5/v6 的 `volatile shared memory` warp-synchronous 写法是教学中常见的旧模式。面向现代 GPU 时，优先考虑 v8 的 shuffle 写法，或显式加入 `__syncwarp()` 来表达同步意图。

所有 v0-v8 都是第一阶段 partial reduction。若最终目标是整个数组的单个 sum，需要像 `reduce.cu` 一样做第二阶段，或使用多 kernel、多轮递归、atomic accumulation、cooperative groups，或直接使用 CUB。

## 后续优化建议

优先增加 CUDA 错误检查和边界处理。这是把教学代码推进到可维护 benchmark 的第一步。

把 `THREAD_PER_BLOCK`、每线程元素数、block 数改成模板或命令行参数，做系统 sweep。不同 GPU、不同 N、不同数据类型下，最优点可能不同。

为 v7/v8 增加完整第二阶段归约，把“第一阶段耗时”扩展成“端到端输出一个标量”的耗时。v7/v8 输出 partial sum 少，端到端可能比单 kernel 表格更有优势。

加入 CUB `DeviceReduce::Sum` 作为上界参考。自写 kernel 的价值在于学习和特定场景定制，工程默认方案通常应先和 CUB 对齐。

用 Nsight Compute 观察 `dram__throughput`、`sm__throughput`、`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`、warp stall reason、achieved occupancy。这样可以把“为什么快”从源码推断变成硬件指标验证。
