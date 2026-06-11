# 完整实例：传输 1, 2, 3, 4——从编码到重建的每一个数字

本文用最小的真实例子走完 RS 纠删码全流程：**4 个数据字节 1、2、3、4，
配置 RS(4+2)**（4 数据分片 + 2 校验分片，允许任意丢 2 个）。
每个分片只有 1 个字节——真实场景分片很长，但每个字节位置独立做
同样的运算，看懂 1 个字节就看懂了全部。

文中所有数字都可以用文末的程序复现。背景知识：
[README 算法核心原理](../README.zh-CN.md#算法核心原理)、
[解码内部机制](decoding-internals.zh-CN.md)。

---

## 第 1 步：建码（一次性，与数据无关）

取求值点 0–5 的 Vandermonde 矩阵（第 r 行 = r⁰, r¹, r², r³，
全部在 GF(2^8) 中计算，如第 5 行的 17 = 5²、85 = 5³ 是域乘法结果）：

```
        ┌  1   0   0   0 ┐
        │  1   1   1   1 │
V  =    │  1   2   4   8 │
        │  1   3   5  15 │
        │  1   4  16  64 │
        └  1   5  17  85 ┘
```

右乘顶部 4×4 子阵的逆做系统化，得到编码矩阵：

```
        ┌  1   0   0   0 ┐
        │  0   1   0   0 │   ← 单位阵：数据原样保留（系统码）
A  =    │  0   0   1   0 │
        │  0   0   0   1 │
        │ 27  28  18  20 │   ← 校验行 1
        └ 28  27  20  18 ┘   ← 校验行 2
```

## 第 2 步：编码——传输 6 个字节而不是 4 个

```
p1 = 27·1 ⊕ 28·2 ⊕ 18·3 ⊕ 20·4 = 69
p2 = 28·1 ⊕ 27·2 ⊕ 20·3 ⊕ 18·4 = 94
```

（`·` 是 GF(2^8) 查表乘法，`⊕` 是异或。）实际发送
**[1, 2, 3, 4, 69, 94]**——前 4 个就是原数据，不丢包时直接读，
零解码开销。

## 第 3 步：丢 2 个，照样恢复

假设 **2 和 4 在传输中丢了**，收到 `[1, ?, 3, ?, 69, 94]`。
接收端知道丢的是 1 号和 3 号位置（纠删的前提：位置已知）。

取 A 中现存分片对应的第 0、2、4、5 行组成方阵 A′，
高斯消元求逆（运行时现算，见解码内部机制一文）：

```
       ┌  1   0   0   0 ┐                ┌   1    0    0    0 ┐
A′ =   │  0   0   1   0 │     A′⁻¹ =     │ 143  211  142  211 │
       │ 27  28  18  20 │                │   0    1    0    0 │
       └ 28  27  20  18 ┘                └ 179  143  201  244 ┘
```

用 A′⁻¹ 乘以收到的值 S = (1, 3, 69, 94)：

```
d2 = 143·1 ⊕ 211·3 ⊕ 142·69 ⊕ 211·94 = 2   ✓ 丢的 2 回来了
d4 = 179·1 ⊕ 143·3 ⊕ 201·69 ⊕ 244·94 = 4   ✓ 丢的 4 回来了
```

（d1、d3 对应单位行，直接透传现存值。）

## 值得注意的三点

1. **冗余去哪了**：69 和 94 各自"压缩"着全部 4 个数据的信息
   （校验行系数全非零），所以任何 2 个位置的丢失都能补；
2. **真实分片只是按字节重复**：分片 1 MB 时，同一个 A′⁻¹ 对 100 万个
   字节位置逐个做上述乘加——这就是热点集中在 `mul_slice_xor`、
   SIMD 优化有效的原因；
3. **LRU 缓存缓存的就是 A′⁻¹**：下一个同样丢 1、3 号分片的条带
   直接复用，免去 O(k³) 求逆。

---

## 校验行的数字 [27 28 18 20] 是怎么来的？

> 想进一步拆解 V、top、top⁻¹ 各自的身份、"右乘"的确切含义以及
> top⁻¹ 的高斯-约当求逆全过程，见姊妹篇
> [encoding-matrix-anatomy.zh-CN.md](encoding-matrix-anatomy.zh-CN.md)。

### 机械层：V 的第 4、5 行 × top⁻¹

校验行就是 Vandermonde 的第 4、5 行——求值点 4 和 5 的幂次
`[1,4,16,64]`、`[1,5,17,85]`——右乘 top⁻¹ 的结果。
"随机感"来自 top⁻¹：求逆过程中的域乘法不断触发
[x⁸ 回绕规则](gf2-16-and-primitive-poly.zh-CN.md#回绕规则的数学推导x⁸--x⁴x³x²1)。

### 语义层：拉格朗日基函数在点 4、5 处的取值

更深一层（可程序验证）：**A[4][i] 恰好等于拉格朗日基多项式 Lᵢ(x)
在 x=4 处的值**。也就是说，编码的真正含义是：

> 找到过 (0,d₁), (1,d₂), (2,d₃), (3,d₄) 的唯一 3 次多项式 P(x)，
> 校验字节就是它在 x=4、x=5 处的采样：p₁ = P(4)，p₂ = P(5)。

这正是 RS"多项式采样"第一性原理的直接体现——6 个采样点里任取 4 个
都能还原这条 3 次曲线。拉格朗日公式给出闭式（⊖ 在 GF(2^8) 中即异或）：

```
Lᵢ(4) = ∏_{j≠i} (4 ⊖ j) / (i ⊖ j)
```

手算第一个数 **27**（i=0）：

```
L₀(4) = (4⊕1)(4⊕2)(4⊕3) / [(0⊕1)(0⊕2)(0⊕3)]
      = (5 · 6 · 7) / (1 · 2 · 3)
      = 90 / 6          （5·6=30，30·7=90；1·2·3=6，均为域运算）
      = 27
```

程序验证（文末程序输出）：

```
L_i(4): [27 28 18 20]   ← 与校验行 1 完全一致
L_i(5): [28 27 20 18]   ← 与校验行 2 完全一致
P(4)=69, P(5)=94        ← 与编码结果完全一致
```

### 数值层：为什么是 27 而不是别的

拉格朗日公式里的乘除全在 GF(2^8) 中进行，结果依赖本原多项式 0x11D
的回绕规则——换一个本原多项式，同样的公式会算出另一组数。
这再次解释了为何所有 RS 实现必须统一 0x11D 才能互操作。

### 彩蛋：两行的对称性

`[27 28 18 20]` 与 `[28 27 20 18]` 恰好两两互换：因为求值点 4 和 5
只差最低位（4⊕5=1），而插值点 0,1 与 2,3 也各自只差最低位，
异或结构让 L₀/L₁ 和 L₂/L₃ 在两个点上正好换位。

---

## 复现程序

### 程序一：完整编码/重建流程

```cpp
#include <cstdio>
#include <optional>
#include <vector>
#include <rse/rse.hpp>

namespace gf8 = rse::gf8;
using F = gf8::Field;
using M = rse::Matrix<F>;

void print_matrix(const char* name, const M& m) {
    std::printf("%s =\n", name);
    for (std::size_t r = 0; r < m.row_count(); ++r) {
        std::printf("  [");
        for (std::size_t c = 0; c < m.col_count(); ++c)
            std::printf(" %3d", m.get(r, c));
        std::printf(" ]\n");
    }
}

int main() {
    // 建码（与 ReedSolomon::create(4,2) 内部完全相同）
    const M vander = M::vandermonde(6, 4);
    const M top = vander.sub_matrix(0, 0, 4, 4);
    const M A = vander.multiply(top.invert().value());
    print_matrix("编码矩阵 A", A);

    // 编码
    const auto r = rse::galois_8::ReedSolomon::create(4, 2).value();
    std::vector<std::vector<std::uint8_t>> shards = {{1}, {2}, {3}, {4}, {0}, {0}};
    r.encode(shards).value();
    std::printf("校验: [%d %d]\n", shards[4][0], shards[5][0]);

    // 丢分片 1、3 后重建
    std::vector<std::optional<std::vector<std::uint8_t>>> opt(shards.begin(), shards.end());
    opt[1] = std::nullopt;
    opt[3] = std::nullopt;
    r.reconstruct(opt).value();
    std::printf("重建: [%d %d %d %d]\n",
                opt[0]->at(0), opt[1]->at(0), opt[2]->at(0), opt[3]->at(0));

    // 展示重建内部的 A' 及其逆
    M sub(4, 4);
    const std::size_t alive[] = {0, 2, 4, 5};
    for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 4; ++c) sub.set(i, c, A.get(alive[i], c));
    print_matrix("A'", sub);
    print_matrix("A'^-1", sub.invert().value());
    return 0;
}
```

### 程序二：校验行 = 拉格朗日基函数取值的验证

```cpp
#include <cstdio>
#include <rse/galois_8.hpp>
namespace gf8 = rse::gf8;

// 拉格朗日基函数 L_i 在 x 处的值（插值点为 0..k-1）
std::uint8_t lagrange(int i, int x, int k) {
    std::uint8_t num = 1, den = 1;
    for (int j = 0; j < k; ++j) {
        if (j == i) continue;
        num = gf8::mul(num, static_cast<std::uint8_t>(x ^ j));
        den = gf8::mul(den, static_cast<std::uint8_t>(i ^ j));
    }
    return gf8::div(num, den);
}

int main() {
    std::printf("L_i(4): [%d %d %d %d]\n",
                lagrange(0,4,4), lagrange(1,4,4), lagrange(2,4,4), lagrange(3,4,4));
    std::printf("L_i(5): [%d %d %d %d]\n",
                lagrange(0,5,4), lagrange(1,5,4), lagrange(2,5,4), lagrange(3,5,4));

    // 编码语义验证: p1 = P(4)，P 为过 (0,1)(1,2)(2,3)(3,4) 的 3 次多项式
    std::uint8_t p1 = 0, p2 = 0;
    const std::uint8_t d[] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) {
        p1 ^= gf8::mul(lagrange(i,4,4), d[i]);
        p2 ^= gf8::mul(lagrange(i,5,4), d[i]);
    }
    std::printf("P(4)=%d, P(5)=%d\n", p1, p2);
    return 0;
}
```

编译运行（在仓库根目录）：

```sh
g++ -std=c++23 -O2 -Iinclude 程序.cpp build/libreed_solomon_erasure.a -o demo && ./demo
```
