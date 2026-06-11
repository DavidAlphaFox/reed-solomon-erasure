# Rabin 不可约测试详解

Rabin 测试是"找本原多项式"流水线里的第一道判定（第二道是本原性验幂，
见 [gf2-16-and-primitive-poly.zh-CN.md](gf2-16-and-primitive-poly.zh-CN.md)）。
本文从数学基石讲到可复跑的完整实现，并用 8 次多项式的全空间枚举
交叉验证理论计数。

---

## 1. 要解决的问题

给定 GF(2) 上的 n 次多项式 f(x)，判定它是否**不可约**（不能分解成
两个低次多项式的乘积）。朴素做法是试除所有次数 ≤ n/2 的多项式，
候选有 O(2^(n/2)) 个——指数爆炸。Rabin 测试把它变成**多项式时间**的判定。

## 2. 数学基石：一个漂亮的因式分解定理

整个测试建立在有限域理论最优雅的定理之一上：

> **x^(2^m) − x 恰好等于"所有次数整除 m 的 GF(2) 上首一不可约多项式"
> 的乘积（每个恰好出现一次）。**

为什么：GF(2^m) 的全部 2^m 个元素正是 x^(2^m) − x 的全部根
（Frobenius 映射的不动点）；而每个元素的极小多项式是某个不可约多项式，
其次数必整除 m。反过来，每个次数 d | m 的不可约多项式的根都落在
GF(2^d) ⊆ GF(2^m) 里。两边对账，分解唯一且无重因子。

举例：x⁴ − x = x(x+1)(x²+x+1)——次数整除 2 的不可约多项式恰好是这三个。

## 3. 测试的两个条件

由该定理，**n 次多项式 f 不可约 ⟺ 同时满足**：

```
(a) x^(2ⁿ) ≡ x (mod f)
(b) 对 n 的每个素因子 q：gcd( x^(2^(n/q)) − x,  f ) = 1
```

- **(a) 是"包含"方向**：f 不可约则 f 整除 x^(2ⁿ)−x（定理直接给出），
  即 x^(2ⁿ) ≡ x。它排除了"含有次数不整除 n 的因子"与"含重因子"
  （x^(2ⁿ)−x 无重因子）两种情况；
- **(b) 是"排除"方向**：(a) 通过后，f 的所有不可约因子次数都整除 n。
  若 f 可约，必有某个因子的次数 d 是 n 的**真**因子；而任何真因子 d
  必整除某个 n/q（取 q 为 n/d 的素因子即可）。该因子会同时整除
  x^(2^(n/q))−x 和 f，让 gcd ≠ 1 暴露出来。

关键的效率点在 (b)：**不需检查 n 的所有真因子，只需检查"极大真因子"
n/q**（q 取 n 的素因子）——所有真因子都被这几个 n/q 覆盖。
对 n = 8 = 2³ 只有一个素因子 2，整个测试就两条：
`x^256 ≡ x (mod f)` 与 `gcd(x^16 − x, f) = 1`。

## 4. 怎么算：重复平方 + Frobenius

`x^(2ⁿ) mod f` 的指数大得吓人，实际只是 n 次连续平方：

```
g ← x
重复 n 次:  g ← g² mod f     // 指数逐次翻倍：x², x⁴, x⁸, ..., x^(2ⁿ)
```

每步是"次数 < n 的多项式平方再模 f"。GF(2) 上平方尤其便宜：
交叉项系数为 2 ≡ 0，只需把每个 x^i 映射为 x^2i（比特位散开）。
gcd 用多项式版欧几里得算法。总复杂度 O(n³) 级（用快速乘法可更低）
——n=8 是微秒级，密码学规模的 n 也完全可行。

## 5. 手算最小例子（n = 2）

- **f = x²+x+1**：(a) x² ≡ x+1，x⁴ ≡ (x+1)² = x²+1 ≡ x ✓；
  (b) n/q = 1，gcd(x²−x, f)：x²+x ≡ 1 (mod f)，gcd = 1 ✓ → **不可约**；
- **f = x²+1**（= (x+1)²，可约）：x² ≡ 1，x⁴ ≡ 1 ≠ x → (a) 失败，
  重因子被 x^(2ⁿ)−x 的无重因子性质卡住 ✓。

## 6. 程序实证：跑一遍 8 次多项式的全空间

完整实现（GF(2)[x] 用位串表示，可直接编译运行）：

```cpp
#include <cstdint>
#include <cstdio>
#include <utility>

using Poly = std::uint32_t;          // 第 i 位 = x^i 的系数

int degree(Poly p) { return p ? 31 - __builtin_clz(p) : -1; }

Poly poly_mod(Poly a, Poly f) {      // a mod f
    for (int d = degree(a); d >= degree(f); d = degree(a))
        a ^= f << (d - degree(f));
    return a;
}

Poly poly_mulmod(Poly a, Poly b, Poly f) {   // a*b mod f
    Poly r = 0;
    while (b) { if (b & 1) r ^= a; b >>= 1; a = poly_mod(a << 1, f); }
    return poly_mod(r, f);
}

Poly poly_gcd(Poly a, Poly b) { while (b) { a = poly_mod(a, b); std::swap(a, b); } return a; }

// Rabin 测试，n = 8（素因子只有 2）：
//   (a) x^(2^8) ≡ x (mod f)        —— 8 次连续平方
//   (b) gcd(x^(2^4) - x, f) = 1    —— 4 次连续平方后求 gcd
bool rabin_irreducible_deg8(Poly f) {
    Poly g = 0b10;                                   // g = x
    for (int i = 0; i < 4; ++i) g = poly_mulmod(g, g, f);
    if (poly_gcd(g ^ 0b10, f) != 1) return false;    // (b) x^16 - x
    for (int i = 0; i < 4; ++i) g = poly_mulmod(g, g, f);
    return g == 0b10;                                // (a) x^256 == x
}

// 本原性：2 的阶 = 255 ⟺ 对 255 的素因子 3,5,17，x^(255/p) != 1
bool primitive_deg8(Poly f) {
    if (!rabin_irreducible_deg8(f)) return false;
    auto pw = [&](int e) { Poly r = 1, b = 0b10;
        while (e) { if (e & 1) r = poly_mulmod(r, b, f);
                    b = poly_mulmod(b, b, f); e >>= 1; } return r; };
    return pw(85) != 1 && pw(51) != 1 && pw(15) != 1;
}

int main() {
    std::printf("0x11D: 不可约=%d 本原=%d\n",
                rabin_irreducible_deg8(0x11D), primitive_deg8(0x11D));
    std::printf("0x11B: 不可约=%d 本原=%d\n",
                rabin_irreducible_deg8(0x11B), primitive_deg8(0x11B));
    std::printf("0x11C: 不可约=%d\n", rabin_irreducible_deg8(0x11C));

    int irreducible = 0, primitive = 0;
    for (Poly f = 0x100; f < 0x200; ++f) {           // 全部 256 个 8 次多项式
        irreducible += rabin_irreducible_deg8(f);
        primitive += primitive_deg8(f);
    }
    std::printf("不可约 %d 个，本原 %d 个\n", irreducible, primitive);
    return 0;
}
```

运行结果：

```
0x11D: 不可约=1 本原=1        ← 本库所用
0x11B: 不可约=1 本原=0        ← AES 多项式：不可约但不本原
0x11C: 不可约=0               ← 可约反例
不可约 30 个，本原 16 个
```

与理论计数完全吻合：

- 8 次不可约多项式数 = (1/8)·Σ_{d|8} μ(d)·2^(8/d) = (2⁸ − 2⁴)/8 = **30**
  （Möbius 计数公式）；
- 其中本原的 = φ(255)/8 = φ(3·5·17)/8 = 128/8 = **16**。

0x11B "不可约但不本原"与
[gf2-16-and-primitive-poly.zh-CN.md](gf2-16-and-primitive-poly.zh-CN.md)
中"0x11B 域中 2 的阶 = 51"的实验互相印证。

## 7. 几点收尾说明

- **流水线位置**：Rabin 测试是第一道过滤（不可约性），通过后再做
  第二道——对 2ⁿ−1 的素因子验幂判本原。两道都是确定性判定。
- **与 Miller-Rabin 的关系**：同一位 Michael O. Rabin，但别混淆——
  Miller-Rabin 素性测试是**概率性**的，Rabin 不可约测试是**确定性**的，
  结论无条件正确。实践中的"随机性"只出现在外层：随机抽候选多项式
  逐个测（不可约密度约 1/n，期望试 n 次命中）。
- **工程上很少亲手跑它**：低次多项式文献早已枚举成表；只有需要
  非标准参数（如本库 GF(2^16) 扩域的 EXT_POLY，sage 的
  `irreducible_element` 内部就是"枚举 + Rabin"）才现场搜索。
- **对照记忆**：条件 (a) 要求 f 整除 x^(2ⁿ)−x（"是自己人"），
  条件 (b) 要求 f 与低层的 x^(2^(n/q))−x 互素（"不混入低次因子"）。
  一进一出，恰好钉死"不可约"。
