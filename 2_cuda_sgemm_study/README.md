# 2_cuda_sgemm_study 优化报告

本文记录 `2_cuda_sgemm_study` 目录中 CUDA SGEMM 示例的优化路线。主线文件从 `my_sgemm_v0_global_memory.cu` 迭代到 `my_sgemm_v8_double_buffer.cu`，目标是计算单精度矩阵乘法：

`C[M, N] = A[M, K] x B[K, N]`

当前所有示例都固定使用 row-major `float` 矩阵，`main()` 中的测试规模统一为 `M = N = K = 512`，并用 CPU 版本做结果校验。和 `1_cuda_reduce_study` 一样，这份报告重点不是只给出“哪个版本更快”，而是梳理每一步到底减少了什么开销、又引入了什么新代价。

## 结论摘要

这组 SGEMM 的优化主线非常清晰：先从 naive 的 global memory 点积版本出发，逐步把数据搬到 shared memory 做 tile 复用，再让每个线程一次计算更多输出元素，用 `float4` 提高向量化搬运效率，接着引入寄存器分块和 outer product 累加，最后尝试 shared memory 转置和 double buffer 流水线。

当前源码在本机 `GTX 1650 / sm_75`、`512^3` 这个较小问题规模下，**最快版本不是最终的 v8，而是 v6**。这说明 SGEMM 优化并不是“技巧越高级越快”，而是要看 tile 形状、寄存器压力、`__syncthreads()` 频率、共享内存布局和目标 GPU 的平衡点。对这份代码来说，`v6` 的 `64x64x64` tile、`4x4` thread tile 和部分 `float4` 化已经达到了较好的折中；`v7/v8` 虽然概念上更先进，但在当前问题规模和硬件上反而出现回退。

另一个必须单独指出的点是：`v1` 不是一个“略慢的版本”，而是一个**不可运行的实验版本**。它试图把 `16 x K` 的 A panel 和 `K x 16` 的 B panel 一次性全部放进 shared memory，在 `K = 512` 时需要约 `64 KiB` shared memory，已经超出常见默认上限，所以 `main()` 中连 kernel launch 都被直接注释掉了。它的意义是说明“shared memory 要分块滑窗使用，而不是把整条 K 维一次性塞进去”。

## 测试环境与口径

测试时间：`2026-05-01`。

测试 GPU：`NVIDIA GeForce GTX 1650`，Driver `552.22`，Compute Capability `7.5`。

编译环境：CUDA 编译器 `12.4.99`，Host 编译器 `GCC 11.4.0`。

编译口径：单独构建 `Release + sm_75`，避免沿用仓库中现有 `build` 目录里不匹配当前 GPU 架构的配置。

```bash
cmake -S . -B /tmp/cuda_sgemm_perf_build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build /tmp/cuda_sgemm_perf_build -j 8
```

运行口径：由于受限 sandbox 中无法访问 CUDA compute 设备，程序在沙箱内会表现为 `0.000 ms / inf TFLOPS / 输出全零`。下面数据来自沙箱外的真实 GPU 运行结果。

输入规模：所有版本都使用 `M = N = K = 512`。

正确性口径：可运行版本均与 CPU 结果比较，最大误差约为 `1.1e-5`；`v1` 未实际执行 kernel，因此输出错误。

性能口径：代码同时打印“有效带宽”和“TFLOPS”。对于 SGEMM 来说，**TFLOPS 比带宽更有参考意义**，因为源码里的带宽公式只把 `A/B/C` 的逻辑大小各记了一次，没有统计真实的 DRAM 读写次数，也无法反映 tile 复用带来的数据重用效果。

测量注意：当前每个程序只做了单次 kernel 计时，没有预热，也没有多轮平均；`512^3` 本身也不算大，所以表中数据更适合看趋势，而不是拿来和 cuBLAS 做严格对标。

## 实测性能

`v1` 的 kernel launch 在源码里被注释掉，因此不参与性能排名。其 `0.007 ms` 只是空事件窗口，不代表任何真实算子性能。

| 版本 | 文件 | 核函数耗时 | 计算性能 | 相对 v0 加速 | 状态 |
|---|---|---:|---:|---:|---|
| v0 | `my_sgemm_v0_global_memory.cu` | `2.758 ms` | `0.097 TFLOPS` | `1.00x` | 正确 |
| v1 | `my_sgemm_v1_shared_memory.cu` | `N/A` | `N/A` | `N/A` | 共享内存超限，kernel 未执行 |
| v2 | `my_sgemm_v2_shared_memory_sliding_windows.cu` | `2.092 ms` | `0.128 TFLOPS` | `1.32x` | 正确 |
| v3 | `my_sgemm_v3_increase_work_of_per_thread.cu` | `2.603 ms` | `0.103 TFLOPS` | `1.06x` | 正确 |
| v4 | `my_sgemm_v4_using_float4.cu` | `2.061 ms` | `0.130 TFLOPS` | `1.34x` | 正确 |
| v5 | `my_sgemm_v5_register_outer_product.cu` | `1.777 ms` | `0.151 TFLOPS` | `1.55x` | 正确 |
| v6 | `my_sgemm_v6_register_outer_product_float4.cu` | `1.568 ms` | `0.171 TFLOPS` | `1.76x` | 正确 |
| v7 | `my_sgemm_v7_a_smem_transpose.cu` | `1.872 ms` | `0.143 TFLOPS` | `1.47x` | 正确 |
| v8 | `my_sgemm_v8_double_buffer.cu` | `2.397 ms` | `0.112 TFLOPS` | `1.15x` | 正确 |

从结果看，性能提升最明显的阶段是：

1. `v0 -> v2`：从纯 global memory 点积切换到标准 shared memory tiling。
2. `v4 -> v6`：从“仅仅用 `float4` 搬运”进化到“寄存器分块 + outer product + 更大 tile”。
3. `v6 -> v7/v8`：技巧继续增加，但不再带来正收益，说明瓶颈已经从“数据搬运太原始”转向“同步、寄存器占用和 tile 选择是否平衡”。

## 版本路线总览

| 版本 | Block 配置 | Block 输出 tile | 每线程输出 | 主要新增优化 | 主要问题 |
|---|---|---|---|---|---|
| v0 | `16 x 16` | `16 x 16` | `1 x 1` | naive 点积 | A/B 全靠 global memory 反复读取 |
| v1 | `16 x 16` | `16 x 16` | `1 x 1` | 想一次性缓存完整 panel | shared memory 超限，方案不可执行 |
| v2 | `16 x 16` | `16 x 16` | `1 x 1` | `BK = 16` 滑窗 tiling | 每线程仍只算 1 个输出，寄存器复用少 |
| v3 | `16 x 16` | `32 x 32` | `2 x 2` | 增加每线程工作量 | 访问分散，额外寄存器/索引开销大 |
| v4 | `8 x 32` | `32 x 32` | `1 x 4` | `float4` 向量化 load/store | 仍是点积写法，计算侧复用一般 |
| v5 | `8 x 32`，逻辑重排后 `16 x 16` | `32 x 32` | `2 x 2` | outer product + 寄存器分块 | tile 规模仍偏小 |
| v6 | `16 x 16` | `64 x 64` | `4 x 4` | 更大 tile，shared 到寄存器部分 `float4` 化 | A 侧 shared 读取仍不是最顺 |
| v7 | `16 x 16` | `64 x 64` | `4 x 4` | A 在 shared 中转置，便于 `float4` 读取 | 转置代价没有被收益覆盖 |
| v8 | `16 x 16` | `128 x 128`，`BK = 8` | `8 x 8` | double buffer 流水线 | `BK` 太小、同步变多、寄存器压力偏高 |

## 背景知识

### SGEMM 的核心不是“乘加本身”，而是数据复用

矩阵乘法的单个输出元素 `C[i, j]` 需要一条长度为 `K` 的点积。如果直接按 `v0` 的写法做，每个线程都独立去读 `A[i, :]` 和 `B[:, j]`，那么同一行 A 和同一列 B 会被大量重复读取。

SGEMM 优化的本质就是把这些重复读取变成“先搬一小块到更快的存储层，再让很多线程复用这块数据”：

1. 先做 block 级 tile，把 `A` 的 `BM x BK` 和 `B` 的 `BK x BN` 搬到 shared memory。
2. 再做 thread 级 tile，让每个线程一次计算 `RM x RN` 个输出元素。
3. 对搬运路径做向量化，比如 `float4`，减少指令数和地址计算。
4. 对 shared memory 布局做转置，让计算阶段的读取尽量连续。
5. 在可能的时候做双缓冲，让“搬下一块”和“算当前块”交替推进。

### GEMM 常见瓶颈

这组代码很适合观察几个典型矛盾：

- 共享内存变大后，数据复用更多，但 occupancy 可能下降。
- 每线程计算更多输出后，算术强度更高，但寄存器压力也会上升。
- `float4` 能减少访存指令，但前提是数据对齐且线程映射正确。
- double buffer 在概念上很好，但如果 `BK` 选得太小、同步太频繁，反而会变慢。

因此 SGEMM 优化不能只盯着“global -> shared -> reg”这条方向，还必须同时平衡：

- `BM/BN/BK` 这三个 tile 维度。
- `RM/RN` 这两个 thread tile 维度。
- shared memory 使用量。
- 寄存器数量。
- `__syncthreads()` 的次数。
- warp 访问是否连续、是否方便向量化。

## 迭代详解

### v0：global memory 直接点积

文件：`my_sgemm_v0_global_memory.cu`。

这是最朴素的写法：`16 x 16` 的 block 覆盖 `C` 的一个 `16 x 16` 子块，每个线程只负责一个输出元素 `C[y, x]`，然后在 `K` 维上做完整点积。

优点是直观，线程到输出元素的映射也很简单；缺点同样直接：A 的某一行会被同一个 block 内多个线程重复读取，B 的某一列也一样，几乎没有任何片上复用。当前实测 `2.758 ms / 0.097 TFLOPS`，是整条路线的 baseline。

### v1：错误的共享内存使用方向

文件：`my_sgemm_v1_shared_memory.cu`。

这个版本的想法是“既然 global memory 慢，那就把当前 block 需要的 A 和 B 全部搬到 shared memory”。问题在于它选择了整条 K 维：

- `a_shared[BLOCK_SIZE][BK] = 16 x 512`
- `b_shared[BK][BLOCK_SIZE] = 512 x 16`

两块 shared memory 合起来是：

`16 x 512 x 4 + 512 x 16 x 4 = 65536 bytes`

也就是约 `64 KiB`。这对大多数默认 shared memory 配置来说已经过大，所以源码里直接把 kernel launch 注释掉了。

这个版本的价值在于说明：**shared memory 应该缓存的是 `BK` 小块，而不是整条 K 维 panel**。真正可行的方向不是“一次搬完整个 K”，而是像 v2 那样做滑动窗口。

### v2：标准 shared memory tiling

文件：`my_sgemm_v2_shared_memory_sliding_windows.cu`。

这是第一版真正可用的 tiled GEMM。它把 block 保持为 `16 x 16`，然后令 `BK = 16`，每次只把：

- `A` 的 `16 x 16`
- `B` 的 `16 x 16`

搬入 shared memory，做完这一小段 K 累加后再继续处理下一段。

它解决了 `v0` 最大的问题：同一个 tile 中的 A/B 数据能被整个 block 复用，而不是每个线程各自从 global memory 重新取一遍。实测从 `2.758 ms` 降到 `2.092 ms`，达到了 `1.32x` 加速，是第一步真正有效的优化。

它的局限也很明确：每个线程仍然只算 `1 x 1` 输出，shared memory 里的数据虽然已经被 block 复用，但 thread 级别的寄存器复用还很弱。

### v3：增加每线程工作量，但映射还不够理想

文件：`my_sgemm_v3_increase_work_of_per_thread.cu`。

这个版本把 block 仍然保持为 `16 x 16` 个线程，但让每个 block 处理 `32 x 32` 的输出 tile。做法是引入：

- `BLOCK = 16`
- `STRIDE = 2`
- `STEP = 32`

于是每个线程不再只负责一个输出，而是负责 `2 x 2` 个输出元素，保存在 `sum[2][2]` 中。

思路本身是正确的：让每个线程做更多工作，可以提高 arithmetic intensity，减少“取一次数据只算一个结果”的浪费。但当前实现里，这 `2 x 2` 输出并不是连续的一小块，而是分散在 `32 x 32` tile 的四个象限上；同时，shared memory 也扩大成了 `32 x 32`。

在当前 GPU 和问题规模下，这一步没有带来收益，反而从 `2.092 ms` 回退到 `2.603 ms`。这说明“增加每线程工作量”本身不是充分条件，**线程映射、输出布局和后续向量化友好性同样关键**。

### v4：引入 `float4`，让线程处理连续的 4 个输出

文件：`my_sgemm_v4_using_float4.cu`。

`v3` 的问题之一是每线程负责的 4 个输出是离散的，不利于向量化。`v4` 重新设计了线程块形状：

- block 改为 `8 x 32`
- block 输出 tile 仍是 `32 x 32`
- 每个线程负责同一行上的连续 `4` 个输出

这样一来，global -> shared 的搬运可以直接使用：

`FETCH_FLOAT4(...) = FETCH_FLOAT4(...)`

`A` 和 `B` 都按连续 `float4` 读取，输出写回时每个线程也是连续列方向的 4 个值。这个版本的重点不是改变算法结构，而是**让访存模式更适合 GPU 向量化搬运**。

实测 `2.061 ms / 0.130 TFLOPS`，略优于 `v2`，说明单纯把 global memory 访问做顺、做宽，已经能带来稳定收益。

### v5：从点积切换到 outer product，并开始做寄存器分块

文件：`my_sgemm_v5_register_outer_product.cu`。

这是这条路线里最关键的一步之一。前面的版本，本质上还是“每个输出元素各做各的点积”；`v5` 开始转向更常见的高性能 GEMM 写法：**让每个线程在寄存器里维护一个小的 `RM x RN` 输出块，并按 K 维逐步做 outer product 累加**。

当前参数是：

- `BM = BN = BK = 32`
- `RM = RN = 2`
- block 物理形状仍是 `8 x 32`

但代码把 `8 x 32 = 256` 个物理线程重新映射成了逻辑上的 `16 x 16` 网格，使得每个线程正好负责一个连续的 `2 x 2` 输出子块。这样每次从 shared memory 取出：

- 一条 `2 x 1` 的 A 向量
- 一条 `1 x 2` 的 B 向量

就能在寄存器里更新一个 `2 x 2` 的临时块。

这一步的意义很大：同样一次 shared memory 读取，不再只服务于一个结果，而是服务于一个寄存器小块。实测也明显变快，来到 `1.777 ms / 0.151 TFLOPS`。

### v6：扩大 tile，并让计算阶段也部分 `float4` 化

文件：`my_sgemm_v6_register_outer_product_float4.cu`。

`v5` 已经证明了“outer product + 寄存器分块”有效，`v6` 继续把这条路做完整：

- block 改为 `16 x 16`
- `BM = BN = BK = 64`
- `RM = RN = RK = 4`
- 每个线程负责 `4 x 4` 共 16 个输出

这个版本有两个重要变化：

1. tile 从 `32 x 32 x 32` 放大到 `64 x 64 x 64`，提高了 block 级数据复用。
2. 计算阶段里，B 从 shared memory 读到寄存器时可以直接 `float4` 取数。

A 侧在 compute 阶段仍然主要是按标量方式从 shared memory 取一列 `4` 个数，但整体结构已经相当接近经典的 register blocked GEMM。

当前测试下，`v6` 是整套源码中表现最好的版本：`1.568 ms / 0.171 TFLOPS`，相对 `v0` 约 `1.76x`。这说明在当前硬件和矩阵规模下，`64x64x64` tile 加 `4x4` thread tile 是一个不错的平衡点。

### v7：把 A 在写入 shared memory 时转置

文件：`my_sgemm_v7_a_smem_transpose.cu`。

`v6` 的剩余问题在于：虽然 B 可以从 shared memory 里连续 `float4` 取数，但 A 侧在计算阶段还不够顺。`v7` 的改动就是在 global -> shared 时把 A 做一次转置：

- 之前：`shared_a[BM][BK]`
- 现在：`shared_a[BK][BM]`

这样进入计算阶段后，每个线程就可以像读取 B 一样，直接从 shared memory 中连续取出 A 的 `4` 个元素：

- `FETCH_FLOAT4(a_reg[0])`
- `FETCH_FLOAT4(b_reg[0])`

从设计上看，这是在为更完整的 shared->register 向量化铺路，方向没有问题。

但当前实测结果是 `1.872 ms / 0.143 TFLOPS`，反而比 `v6` 慢。原因大概率有三类：

1. A 的转置写入本身引入了额外指令和地址计算。
2. 当前 GPU 和矩阵规模下，这部分额外代价没有被后续连续读取的收益覆盖。
3. tile 没变、同步没变，优化只发生在 shared memory 布局层面，收益空间本来就有限。

所以 `v7` 更适合理解为“布局优化实验”，而不是一定会赢的最终答案。

### v8：double buffer 流水线

文件：`my_sgemm_v8_double_buffer.cu`。

`v8` 试图再往前走一步：在 shared memory 里做双缓冲，让“装载下一块”和“计算当前块”交替推进。

这个版本的参数变化非常大：

- `BM = BN = 128`
- `BK = 8`
- `RM = RN = 8`
- 每个线程负责 `8 x 8 = 64` 个输出
- `shared_a[2][BK][BM]`
- `shared_b[2][BK][BN]`

和 `v6/v7` 相比，它明显是更激进的设计：

- C tile 更大。
- thread tile 更大。
- shared memory 从单缓冲变为双缓冲。
- A 在 shared 中仍保持转置布局。

但当前实测却退回到了 `2.397 ms / 0.112 TFLOPS`，只比 `v0` 好一点点。这个结果并不意外，原因其实很典型：

1. `BK` 从 `64` 降到了 `8`，意味着 K 维循环次数从 `8` 次增加到 `64` 次，循环和同步开销显著增多。
2. 每个线程维护 `8 x 8` 的寄存器临时块，寄存器压力明显增大，容易压低 occupancy。
3. 这版虽然叫 double buffer，但仍然依赖常规 load + `__syncthreads()`，并没有使用更激进的异步拷贝机制，所以流水线收益有限。

因此 `v8` 最大的价值是说明：**double buffer 不是银弹**。如果 `BK` 太小、thread tile 太大、同步仍然很多，那么“看起来更高级”的流水线未必比一个平衡的 `v6` 更快。

## 为什么 v6 最快，而不是 v8

这套代码最值得学习的地方，恰恰是最终结果没有形成“版本号越大越快”的单调曲线。

在当前 `GTX 1650 + 512^3` 规模下，`v6` 胜出的本质是它同时满足了几个平衡条件：

- `64x64x64` tile 已经有足够的数据复用。
- `4x4` thread tile 让每线程工作量明显提升，但还没有把寄存器压力推得太高。
- B 侧 shared->register 已经可以 `float4` 化。
- K 维分块还是 `64`，循环次数不多，同步开销可控。

而 `v7/v8` 没有继续变快，分别暴露了两种常见问题：

- `v7`：只改布局，不改总体算子平衡，收益不足以覆盖转置成本。
- `v8`：优化动作过多，导致更小的 `BK`、更多的循环、更高的寄存器占用，把原先的平衡打破了。

这和真实工程中的 GEMM 调优经验是一致的：**你优化的不是某一个点，而是整条数据路径的平衡**。

## 这套源码还缺什么

如果把这份目录当成“学习用 SGEMM 递进实验”，它已经很有价值；但如果想继续逼近高性能实现，还缺几类重要能力：

1. CUDA 错误检查。当前代码没有检查 `cudaMalloc`、`cudaMemcpy`、kernel launch 和 `cudaEvent` 的返回值，导致在受限环境下会出现 `0.000 ms / inf / wrong` 这种误导性结果。
2. 通用边界处理。`v4-v8` 基本都假设 `M/N/K` 恰好是 tile 的整数倍，且 `float4` 对齐成立，更像教学内核而不是通用库内核。
3. 更稳健的 benchmark 方法。现在是单次计时，没有 warmup、没有多次平均、没有剔除异常值。
4. 更现代的流水线手段。如果后续目标是继续做双缓冲，通常还需要结合更合理的 `BK` 以及更先进的异步拷贝机制，而不是只做手工 ping-pong。

## 最后总结

`2_cuda_sgemm_study` 的主线可以概括为：

`global memory 点积 -> shared memory tiling -> 增加每线程工作量 -> float4 向量化 -> 寄存器分块 outer product -> shared memory 布局优化 -> double buffer 流水线`

但真正决定性能的，不是“用了多少技巧”，而是这些技巧是否在当前 GPU 和当前问题规模下形成了更好的平衡。当前代码最成功的一步是 `v5/v6` 把优化重点从“只是搬得更快”转向“让每次搬来的数据在寄存器里做更多有价值的计算”；而 `v7/v8` 则很好地说明了另一面：**优化路线是探索过程，不是必然单调上升的台阶**。
