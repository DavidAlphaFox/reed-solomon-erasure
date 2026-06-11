# reed-solomon-erasure

Reed-Solomon 纠删码的 C++23 实现。

本仓库源自 Rust crate [`reed-solomon-erasure`](https://github.com/darrenldl/reed-solomon-erasure)
(v6.0.0)，已完整重写为 C++23。C++ 实现与 Rust 版**线缆格式兼容**：相同输入
产生逐字节一致的校验分片（已在 x86-64/AVX2 与 AArch64/NEON 上交叉验证）。
原 Rust 源码保留在 git 历史中。

注意：纠删码不直接检测或纠正错误，它的能力是在冗余度足够时**重建缺失的
分片**。错误检测（如校验和）需要另行实现，重建时把损坏的分片标记为缺失即可。

- [构建与使用](#构建与使用)
- [算法核心原理](#算法核心原理)
- [SIMD 如何加速 RS 算法](#simd-如何加速-rs-算法)
- [性能数据与调优建议](#性能数据与调优建议)

更深入的专题文档见 `doc/`：

- [完整实例：传输 1, 2, 3, 4——从编码到重建的每一个数字](doc/worked-example.zh-CN.md)（含校验行系数 = 拉格朗日基函数取值的推导）
- [编码矩阵的解剖：top、top⁻¹、V 与 A](doc/encoding-matrix-anatomy.zh-CN.md)（插值/求值矩阵的身份、"右乘"的含义、高斯-约当求逆逐步演示）
- [重建（解码）的内部机制：哪些是计算，哪些是查表](doc/decoding-internals.zh-CN.md)
- [本原多项式与生成多项式：程序是如何处理的](doc/polynomials.zh-CN.md)
- [GF(2^16) 的含义，以及如何找到一个域的本原多项式](doc/gf2-16-and-primitive-poly.zh-CN.md)（含"钟面"直观类比与 x⁸ ≡ x⁴+x³+x²+1 回绕规则的推导）
- [Rabin 不可约测试详解](doc/rabin-irreducibility-test.zh-CN.md)

---

## 构建与使用

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # 运行测试套件
cmake --install build           # 安装头文件 + 库 + CMake 包
```

同时产出静态库（`libreed_solomon_erasure.a`）与动态库
（`libreed_solomon_erasure.so`，SOVERSION 6）。CMake 目标：源码内
`rse::static` / `rse::shared`，安装后 `find_package(reed_solomon_erasure)`。

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `RSE_BUILD_TESTS` | `ON` | 构建 doctest 测试套件 |
| `RSE_BUILD_BENCH` | `OFF` | 构建带宽基准（`rse_bandwidth`） |

```cpp
#include <rse/rse.hpp>

auto r = rse::galois_8::ReedSolomon::create(3, 2).value();   // 3 数据 + 2 校验

std::vector<std::vector<std::uint8_t>> shards = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},   // 数据分片
    {0, 0, 0}, {0, 0, 0},              // 校验分片（会被覆盖写入）
};
r.encode(shards).value();
assert(r.verify(shards).value());

// 重建方式一：optional 分片（缺失的置为 nullopt）
std::vector<std::optional<std::vector<std::uint8_t>>> opt(shards.begin(), shards.end());
opt[0] = std::nullopt;
opt[4] = std::nullopt;
r.reconstruct(opt).value();

// 重建方式二：就地重建（present 标志标记哪些分片有效）
bool present[] = {false, true, true, true, true};
shards[0] = {9, 9, 9};                 // 已损坏
r.reconstruct(shards, present).value();
```

错误通过 `std::expected<T, rse::Error>` 返回；Rust 版会 panic 的场景
（切片长度不符、非方阵求逆、除零）对应抛出异常。

---

## 算法核心原理

### 1. 问题模型

把一段数据切成 `k` 个等长的**数据分片**，额外计算出 `m` 个**校验分片**，
共 `n = k + m` 个分片分散存放。Reed-Solomon 纠删码保证：**任意丢失至多
`m` 个分片（无论数据还是校验），都能从剩余的任意 `k` 个分片完整恢复。**
这是信息论意义上的最优（MDS 码）：想容忍 `m` 个丢失，至少要 `m` 份冗余。

### 2. 为什么需要有限域 GF(2^8)

编码本质是解线性方程组，需要加减乘除四则运算。但字节是 8 位整数，
普通整数运算会溢出。解决办法是把每个字节看作**有限域 GF(2^8) 的元素**——
一个恰好有 256 个元素的代数系统，四则运算封闭（结果永远还是一个字节）、
每个非零元素都有乘法逆元（除法总是可行）。

GF(2^8) 的元素是系数取 {0,1} 的 7 次以下多项式（即一个字节的 8 个位），
运算规则：

- **加法 = 减法 = 按位异或**（系数模 2 相加），所以 `a + a = 0`；
- **乘法** = 多项式乘法后模一个 8 次不可约多项式（本库与 Rust/Go/Java 各版
  实现一致，使用 `x^8 + x^4 + x^3 + x^2 + 1`，即代码中的生成多项式 29）。

工程上不会真去做多项式乘法。GF(2^8) 的非零元素构成循环群：取生成元
g = 2，每个非零元素都是 g 的某次幂。于是预生成两张表：

```
LOG_TABLE[a] = log_g(a)        EXP_TABLE[n] = g^n
a * b = EXP_TABLE[LOG_TABLE[a] + LOG_TABLE[b]]      （对数相加）
a / b = EXP_TABLE[LOG_TABLE[a] - LOG_TABLE[b]]      （对数相减）
```

本库进一步预生成完整的 256×256 = 64 KiB 乘法表 `MUL_TABLE`，运行期乘法
就是一次查表。所有表都在**编译期**由 `consteval` 函数生成
（`include/rse/detail/gf8_tables.hpp`），替代了 Rust 版的 build.rs 代码生成。

`k + m > 256` 时 GF(2^8) 不够用（见下文编码矩阵需要 n 个互不相同的域元素），
本库另提供 GF(2^16)（`rse::galois_16`）：以 GF(2^8) 为基域的二次扩域，
元素是两字节对，分片上限提升到 65536。

### 3. 编码矩阵：Vandermonde + 系统化

把 `k` 个数据分片在同一字节偏移处的值看成一个列向量
`D = (d_1, ..., d_k)ᵀ`，编码就是乘一个 `n × k` 的**编码矩阵** `A`：

```
        ┌ 1 0 ... 0 ┐
        │ 0 1 ... 0 │   ← 上半部分：k×k 单位阵
  A  =  │ ...       │      （编码后数据分片原样保留，"系统码"）
        │ 0 0 ... 1 │
        │ c11 ... c1k │  ← 下半部分：m 行校验系数
        │ ...        │
        └ cm1 ... cmk ┘

  A · D = (d_1, ..., d_k, p_1, ..., p_m)ᵀ     全部 n 个分片
```

矩阵 `A` 必须满足一个关键性质：**任取 k 行都构成可逆方阵**。
构造方法（`ReedSolomon::build_matrix`）：

1. 取 `n × k` 的 **Vandermonde 矩阵** `V`，第 r 行为
   `(a_r^0, a_r^1, ..., a_r^{k-1})`，其中 a_r 是互不相同的域元素
   （这就是 n ≤ 域的阶的原因）。Vandermonde 的性质保证任取 k 行行列式非零；
2. `V` 的上半部分不是单位阵，直接用的话数据分片编码后会"变样"。
   做系统化变换：`A = V · (V 顶部 k×k 子阵)⁻¹`。右乘可逆阵不改变
   "任取 k 行可逆"的性质，而顶部恰好变成单位阵。

### 4. 编码、校验与重建

**编码**（`encode`）：对每个字节偏移并行地算 `P = C · D`（C 是校验行）。
实现上不按"偏移"循环，而是按分片循环，把内层做成整条切片的批量运算：

```
for 每个数据分片 d_i:
    for 每个校验分片 p_j:
        p_j[0..len] ^= C[j][i] * d_i[0..len]     ← mul_slice_xor，SIMD 热点
```

**校验**（`verify`）：重算一遍校验分片与现有的比对。

**重建**（`reconstruct`）：设丢了若干分片，但仍有 ≥ k 个分片在。
从 `A` 中取出**现存分片对应的 k 行**组成方阵 `A'`，则现存分片满足
`A' · D = S`（S 为现存分片值）。由"任取 k 行可逆"，解出：

```
D = A'⁻¹ · S        （高斯-约当消元求逆，见 matrix.hpp）
```

数据分片齐了之后，缺失的校验分片按正常编码重算。矩阵求逆是 O(k³)，
但只跟"缺了哪些分片"有关、跟数据量无关，本库用 LRU 缓存
（容量 254，键为缺失下标列表）把同一缺失模式的求逆开销摊到一次。

---

## SIMD 如何加速 RS 算法

### 1. 热点在哪里

编码/重建的全部时间几乎都耗在一个原语上：

```
mul_slice_xor(c, in, out):   out[i] ^= c * in[i]，i = 0..len
```

即"整条切片乘以常数 c 再异或累加"。编码做 `k × m` 次这样的整条运算。
标量实现是逐字节查 64 KiB 乘法表——每字节一次访存依赖，吞吐约 1-2 GB/s。

### 2. 核心技巧：半字节拆分 + 并行查表

SIMD 没有"按字节查 256 项表"的指令，但 x86 的 `PSHUFB` 和 ARM 的 `TBL`
可以做**16 字节并行的 16 项查表**：以向量中每个字节的低 4 位为下标，
从一个装在向量寄存器里的 16 项表中取值，一条指令完成 16/32/64 路查表。

256 项的乘法表怎么塞进 16 项？利用 GF(2^n) 乘法对加法（异或）的**分配律**。
把字节 b 拆成高低两个半字节：`b = hi·16 ⊕ lo`，则：

```
c * b = c * (hi·16 ⊕ lo) = (c * hi·16) ⊕ (c * lo)
```

于是对每个乘数 c 预生成两张 16 项小表（编译期完成，见 gf8_tables.hpp）：

```
LOW[c][x]  = c * x         （x = 0..15，低半字节的乘积）
HIGH[c][x] = c * (x << 4)  （高半字节的乘积）
```

内核每个向量宽度（16/32/64 字节）只需 8 条指令（`src/gf8_simd_kernel.hpp`）：

```
in      = load(输入)
lo      = in & 0x0F                  ← 低半字节
hi      = (in >> 4) & 0x0F           ← 高半字节
result  = PSHUFB(LOW[c], lo) ⊕ PSHUFB(HIGH[c], hi)
result ⊕= load(输出)                  ← 仅 mul_xor 变体
store(输出, result)
```

一条 AVX2 指令并行完成 32 个字节的"查表乘法"，这是标量逐字节查表的
数量级提升的来源。

### 3. 工程结构

- **算法只写一次**：`gf8_simd_kernel.hpp` 是模板化的内核本体；每个指令集
  （SSSE3 / AVX2 / AVX-512BW / NEON）只提供一组 `Ops` 原语
  （load/store/splat/and/xor/shift/lookup），约 15 行。
- **运行时分发**：每个 ISA 的内核编译在独立翻译单元里（各自带 `-mssse3`
  等编译选项），库本身不需要 `-march`。首次调用时用 CPUID
  （`__builtin_cpu_supports`）按 AVX-512BW > AVX2 > SSSE3 选择；
  AArch64 上 NEON 是基线特性直接使用。一个二进制适配所有机器。
- **尾部处理**：内核只处理向量宽度整数倍的部分，余下尾部由标量查表补齐。
- **宽指令集的细节**：AVX2/AVX-512 的 `VPSHUFB` 在每个 128 位通道内独立
  查表，16 字节小表需要先广播到各通道；`_mm512_shuffle_epi8` 属于
  AVX-512**BW** 扩展，运行时按 `avx512bw` 检测而非 `avx512f`。
- **一个真实的移植教训**：OpenBSD 的 clang 在 AArch64 上默认
  `+strict-align`，会把非对齐向量加载整体标量化成逐字节 `ldrb`，
  性能掉一个数量级。本库对 NEON 翻译单元显式传 `-mno-strict-align`
  （用户态访问普通内存允许非对齐）。

GF(2^16) 没有 SIMD 路径（与 Rust 版一致），批量运算走通用逐元素循环。

---

## 性能数据与调优建议

实测编码带宽（按数据字节计，`bench/bandwidth.cpp`）：

| 平台 | 后端 | 10+3 × 64 KiB |
| --- | --- | --- |
| i9-13900H（单线程） | AVX2 | ~11 GiB/s |
| Raspberry Pi 4 / Cortex-A72 | NEON | ~320 MiB/s |

来自多线程与缓存实验的几条实用结论：

1. **算法完美可并行**：每个字节位置的计算独立，可按字节范围分块并行，
   块间零依赖。但低冗余配置（如 10+3）很快撞内存带宽墙（约 2 倍封顶）；
   高冗余配置（如 50+20）是计算瓶颈，并行收益接近线性。
2. **优先考虑跨条带并行**：`ReedSolomon` 实例是 const 且线程安全的
   （解码矩阵缓存内部加锁），多个线程共享同一实例、各自编码不同的数据块，
   零改动即可扩展。
3. **大配置先做 cache blocking**：当 `分片数 × 分片长度` 超出 L3 时，
   调用方按 16–64 KiB 分块串行调用 `encode_sep`，单线程即可获得 2 倍以上
   提升（这也是 Rust 版历史上 `bytes_per_encode = 32768` 设计的由来）。
4. 几 KiB 的小负载不要并行，调度开销大于收益。

## 许可证

MIT，见 [LICENSE](LICENSE)。实现派生自 Darren Ldl 的 Rust crate、
Nicolas Trangez 与 Klaus Post 的 SIMD 内核，以及 Backblaze 的
Java Reed-Solomon（本编码方案的源头）。
