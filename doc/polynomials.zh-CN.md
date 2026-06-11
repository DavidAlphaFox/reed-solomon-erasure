# 本原多项式与生成多项式：程序是如何处理的

本文澄清两个容易混淆的术语在本库中的角色：定义域的是**本原多项式**
（代码中的 `PRIMITIVE_POLYNOMIAL`），而经典 RS 纠错码意义上的
"生成多项式"——**本库根本没有**。

| 术语 | 层面 | 作用 | 本库 |
|---|---|---|---|
| **本原多项式**（primitive polynomial） | **域**层面 | 定义 GF(2^8) 本身：乘法的模数 | `PRIMITIVE_POLYNOMIAL = 29`，即 x⁸+x⁴+x³+x²+1（0x11D） |
| **生成多项式**（generator polynomial）g(x)=∏(x−αⁱ) | **码**层面 | BCH 视角的 RS 纠错码用它定义码字 | 不存在（矩阵视角不需要） |

命名沿革：Rust 原版把这个常量叫 "GENERATING_POLYNOMIAL"，按编码理论的
标准术语并不准确，C++ 版已更正为 `PRIMITIVE_POLYNOMIAL`。

---

## 本原多项式：只出现在编译期建表的 5 行循环里

整个代码库中，这个多项式唯一被"使用"的地方是 `gen_log_table`
（`include/rse/detail/gf8_tables.hpp`）：

```cpp
std::size_t b = 1;
for (std::size_t log = 0; log < GROUP_ORDER; ++log) {   // 255 次
    result[b] = log;          // 记录 log_2(b) = log
    b <<= 1;                  // b *= 2（乘以生成元 α = 2 = "x"）
    if (FIELD_SIZE <= b) {
        b = (b - FIELD_SIZE) ^ polynomial;   // 模本原多项式归约
    }
}
```

逐行拆解：

- `b <<= 1` 就是多项式乘以 x（域元素乘以 2）；
- 若结果出现 x⁸ 项（b ≥ 256），用本原多项式归约：由
  x⁸+x⁴+x³+x²+1 ≡ 0 得 **x⁸ ≡ x⁴+x³+x²+1**，所以"减去 256 再异或 29"
  （29 = 0x1D 正是多项式的低 8 位）；
- 循环 255 次，把 α=2 的所有幂记进对数表。

**为什么必须"本原"而不仅是"不可约"**：不可约只保证商环是域；
本原额外保证 **α=2 是乘法群的生成元**，即 2 的幂恰好遍历全部 255 个
非零元素——上面的循环才能给每个元素分配到唯一的对数。
反例：AES 用的 0x11B 不可约但**不本原**（2 的阶只有 51），
拿它跑这个循环，51 步后就开始绕圈，表就建错了。
这正是各家 RS 实现（Backblaze / Klaus Post / 本库）都选 0x11D 的原因。

此后多项式**再也不出现**：`EXP_TABLE`、64 KiB 的 `MUL_TABLE`、
SIMD 用的半字节拆分表全部从 log/exp 表派生，模数已经"烤进"表里了。
运行期没有任何多项式归约运算。

---

## 生成多项式：本库不需要——"纠删"与"纠错"两种视角的分水岭

经典 RS **纠错**码（CD、二维码、DVB）采用 BCH 视角：

- 码字定义为生成多项式 g(x) = (x−α⁰)(x−α¹)…(x−α^(2t−1)) 的倍式；
- 编码 = 多项式除法取余附在数据后；
- 解码要算 syndrome、Berlekamp-Massey 求错误位置多项式、Chien 搜索……
  一整套机制，因为**错误位置未知**，要先把位置"解"出来。

而纠删码场景里**擦除位置是已知的**（哪块盘坏了一清二楚），
不需要定位机制，问题退化成"已知位置的线性方程组求解"。
所以本库走求值/矩阵视角：

- 编码矩阵的每行是求值点的幂次 `(a_r⁰, a_r¹, …, a_r^(k−1))`
  （`include/rse/matrix.hpp` 的 `vandermonde`）——只用到**域元素的幂**，
  从不构造 g(x)；
- 重建 = 抽行、求逆、矩阵乘，见
  [decoding-internals.zh-CN.md](decoding-internals.zh-CN.md)。

两种视角定义的是同一族码，但实现机制完全不同；
"生成多项式"是 BCH 视角的概念，在矩阵视角里没有对应物。

---

## 附：GF(2^16) 的第三个多项式

`include/rse/galois_16.hpp` 还有一个 `EXT_POLY = x² + 2x + 128`：
它是 **GF(2^8) 上的不可约多项式**，用来把 GF(2^8) 扩成二次扩域
GF((2^8)²)。与 GF(2^8) 不同，65536² 的乘法表太大无法预生成，
所以这里的模归约是**运行时显式做的**（`Element::reduce_from`：
乘出 x² 项后用 EXT_POLY 消去），求逆则用扩展欧几里得算法现算。

这个多项式是原作者用 sage 的
`R.irreducible_element(2, algorithm="first_lexicographic")` 选出来的
（取字典序第一个，期望归约系数尽量简单）。

寻找/判定这类多项式的一般方法（Rabin 不可约测试、本原性验幂判定）
及程序验证，见 [gf2-16-and-primitive-poly.zh-CN.md](gf2-16-and-primitive-poly.zh-CN.md)。

---

## 总结

| 多项式 | 在哪 | 何时使用 |
|---|---|---|
| 本原多项式 0x11D | `gf8_tables.hpp` 的 `PRIMITIVE_POLYNOMIAL` | 仅编译期建 log 表；之后烤进所有查找表 |
| 生成多项式 g(x) | 不存在 | 纠删码用矩阵视角，无需 BCH 机制 |
| 扩域多项式 EXT_POLY | `galois_16.hpp` | GF(2^16) 元素乘法的运行时归约 |
