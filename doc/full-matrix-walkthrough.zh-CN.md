# 一镜到底：用矩阵运算完整演示 RS(4+2) 全流程

本文是一份"总览式"演示：数据 1,2,3,4，从建码、编码、隐藏曲线、
丢失重建到闭环验证，**全部用矩阵运算完成**（向量也表示为 n×1 矩阵，
连乘法都走库的 `Matrix::multiply`），每一站打印真实数值。

分步细讲见姊妹篇：[worked-example.zh-CN.md](worked-example.zh-CN.md)
（实例分解）、[encoding-matrix-anatomy.zh-CN.md](encoding-matrix-anatomy.zh-CN.md)
（矩阵解剖）。本文程序在文末，可直接编译复现。

---

## 第 1 幕：建码（与数据无关，create(4,2) 时执行一次）

```
V      = [   1   0   0   0 ]      top    = [   1   0   0   0 ]
         [   1   1   1   1 ]               [   1   1   1   1 ]
         [   1   2   4   8 ]               [   1   2   4   8 ]
         [   1   3   5  15 ]               [   1   3   5  15 ]
         [   1   4  16  64 ]
         [   1   5  17  85 ]      （top = V 的前 4 行）

top^-1 = [   1   0   0   0 ]      核对 top*top^-1 = I ✓
         [ 123   1 142 244 ]
         [   0 122 244 142 ]      （高斯-约当消元求出，
         [ 122 122 122 122 ]        过程见矩阵解剖篇 §6）

A = V*top^-1 = [   1   0   0   0 ]
               [   0   1   0   0 ]   ← 前 4 行变成单位阵（系统码）
               [   0   0   1   0 ]
               [   0   0   0   1 ]
               [  27  28  18  20 ]   ← 校验行
               [  28  27  20  18 ]
```

纯"基建"：V 是求值机、top⁻¹ 是插值机、A 是两机合体。
此时数据尚不存在。

## 第 2 幕：编码（数据登场，一次矩阵乘）

```
d = [ 1 ]            A*d = [  1 ]
    [ 2 ]                  [  2 ]   ← 数据原样穿过单位阵部分
    [ 3 ]                  [  3 ]
    [ 4 ]                  [  4 ]
                           [ 69 ]   ← 校验行"吞掉"插值后的产物
                           [ 94 ]
```

发送 6 个数：`[1, 2, 3, 4, 69, 94]`。

## 幕间：隐藏曲线（仅演示，真实代码跳过这一步）

```
c = top^-1 * d = [   1 ]      V*c = [  1 ]
                 [   1 ]            [  2 ]
                 [ 247 ]            [  3 ]
                 [ 245 ]            [  4 ]
                                    [ 69 ]
（隐藏曲线 P(x) =                   [ 94 ]
  1 + x + 247x² + 245x³）
```

两条路结果**逐位一致**——结合律 `V·(top⁻¹·d) = (V·top⁻¹)·d`
的现场兑现。这就是"曲线系数被隐藏"的证据：代码走的是不经过 c
的那条路，c 是见证人而非参与者（原理见矩阵解剖篇）。

## 第 3 幕：丢失与重建（分片 1、3 没了）

从 A 中抽**现存**分片（0,2,4,5 号）对应的行组成 A′——注意这一步
只用了"缺失模式"（哪几行），完全没碰数据值：

```
A' = [   1   0   0   0 ]      S = [  1 ]   ← 现存分片的值
     [   0   0   1   0 ]          [  3 ]
     [  27  28  18  20 ]          [ 69 ]
     [  28  27  20  18 ]          [ 94 ]

A'^-1 = [   1   0   0   0 ]   ← 第 0、2 行是单位行：
        [ 143 211 142 211 ]      现存数据分片直接透传
        [   0   1   0   0 ]
        [ 179 143 201 244 ]

D = A'^-1 * S = [ 1 ]
                [ 2 ]   ← 丢掉的 2 从 (1,3,69,94) 里"解"了回来
                [ 3 ]
                [ 4 ]   ← 丢掉的 4 同上
```

## 第 4 幕：闭环验证

```
A * D = [ 1  2  3  4  69  94 ]ᵀ    与第 2 幕完全一致 ✓
```

拿重建的数据重新编码，结果与最初的 6 个分片逐位相同——这同时
演示了 `verify` 的原理（重算校验比对）和体系的自洽性。

---

## 回味

整场演示只用了**两种矩阵运算**——乘法和求逆；它们底下又只有
**两种域运算**——查表乘法和异或。从"传 4 个数字防丢 2 个"的需求，
到 `[27 28 18 20]` 这些系数的来历，再到生产代码里 SIMD 加速的
字节乘加（`mul_slice_xor`），所有层次在这一屏输出里全部接通。

对应到生产代码的差异只有两点：

1. 第 1 幕在 `ReedSolomon::create` 中执行一次并缓存（`build_matrix`）；
   第 3 幕的 A′⁻¹ 按缺失模式由 LRU 缓存复用；
2. 真实分片很长：同样的矩阵系数对分片的每个字节位置重复执行，
   内层组织成整条切片的 `mul_slice_xor`（SIMD 热点）。

## 复现程序

```cpp
// RS(4+2) 全流程矩阵演示：数据 1,2,3,4
#include <cstdio>
#include <rse/rse.hpp>
using M = rse::Matrix<rse::gf8::Field>;

void pm(const char* name, const M& m) {
    std::printf("%s\n", name);
    for (std::size_t r = 0; r < m.row_count(); ++r) {
        std::printf("  [");
        for (std::size_t c = 0; c < m.col_count(); ++c) std::printf(" %3d", m.get(r, c));
        std::printf(" ]\n");
    }
}

int main() {
    // 第 1 幕：建码
    const M V = M::vandermonde(6, 4);
    pm("V", V);
    const M top = V.sub_matrix(0, 0, 4, 4);
    const M topinv = top.invert().value();
    pm("top^-1", topinv);
    pm("top*top^-1", top.multiply(topinv));
    const M A = V.multiply(topinv);
    pm("A = V*top^-1", A);

    // 第 2 幕：编码
    const M d = M::new_with_data({{1}, {2}, {3}, {4}});   // 4x1 列向量
    const M shards = A.multiply(d);
    pm("A*d (全部分片)", shards);

    // 幕间：隐藏曲线 + 结合律核对
    const M c = topinv.multiply(d);
    pm("c = top^-1*d (隐藏曲线系数)", c);
    pm("V*c (应与 A*d 相同)", V.multiply(c));

    // 第 3 幕：丢分片 1、3 后重建
    const std::size_t alive[] = {0, 2, 4, 5};
    M Ap(4, 4);
    M S(4, 1);
    for (int i = 0; i < 4; ++i) {
        for (int col = 0; col < 4; ++col) Ap.set(i, col, A.get(alive[i], col));
        S.set(i, 0, shards.get(alive[i], 0));
    }
    pm("A'", Ap);
    const M Apinv = Ap.invert().value();
    pm("A'^-1", Apinv);
    pm("D = A'^-1*S (重建结果)", Apinv.multiply(S));

    // 第 4 幕：闭环验证
    pm("A*D (应与最初分片相同)", A.multiply(Apinv.multiply(S)));
    return 0;
}
```

编译运行（仓库根目录）：

```sh
g++ -std=c++23 -O2 -Iinclude demo.cpp build/libreed_solomon_erasure.a -o demo && ./demo
```
