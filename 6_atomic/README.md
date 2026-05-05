# Cuda原子操作


| 操作 | 函数名 | 说明 |
|------|--------|------|
| 加法 | atomicAdd(addr, val) | 加法 |
| 减法 | atomicSub(addr, val) | 减法 |
| 最大值 | atomicMax(addr, val) | 返回两个值的最大值 |
| 最小值 | atomicMin(addr, val) | 返回两个值的最小值 |
| 与 | atomicAnd(addr, val) | 按位与 |
| 或 | atomicOr(addr, val) | 按位或 |
| 异或 | atomicXor(addr, val) | 按位异或 |
| 交换 | atomicExch(addr, val) | 设置新值并返回旧值 |
| 比较交换 | atomicCAS(addr, compare, val) | 如果当前值等于 compare，则设置为 val |