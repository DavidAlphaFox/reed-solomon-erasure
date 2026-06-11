// galois_16.hpp - 有限域 GF(2^16) 的实现。
//
// 移植自 Rust crate 的 src/galois_16.rs。
//
// 更准确地说，这是 GF((2^8)^2) ——以 GF(2^8) 为基域构造的二次扩域：
// 每个元素是一个次数 < 2 的多项式 a*x + b，其中系数 a、b ∈ GF(2^8)，
// 乘法在模不可约多项式 EXT_POLY = x^2 + 2*x + 128 的意义下进行。
// 这样分片总数上限可以从 256 提升到 65536。
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "galois_8.hpp"

namespace rse::gf16 {

// 作为模的不可约多项式：x^2 + a*x + a^7（系数 [1, 2, 128]）。
// 由 sage 脚本 R.irreducible_element(2, algorithm="first_lexicographic")
// 求得（见原 Rust 仓库的 sage/ 目录），选择它是希望归约运算尽量便宜。
inline constexpr std::array<std::uint8_t, 3> EXT_POLY = {1, 2, 128};

// GF(2^16) 的一个元素，表示为 GF(2^8) 上次数 < 2 的多项式，
// 系数按 [高次项, 常数项] 存储（与 Rust 版 Element([u8; 2]) 一致）。
class Element {
public:
    using Repr = std::array<std::uint8_t, 2>;

    constexpr Element() noexcept : repr_{0, 0} {}
    constexpr explicit Element(Repr r) noexcept : repr_(r) {}
    constexpr Element(std::uint8_t hi, std::uint8_t lo) noexcept : repr_{hi, lo} {}

    // 零元（加法单位元）。
    static constexpr Element zero() noexcept { return Element{}; }
    // 常数元素 n（高次项系数为 0）。
    static constexpr Element constant(std::uint8_t n) noexcept { return Element(0, n); }

    [[nodiscard]] constexpr Repr repr() const noexcept { return repr_; }
    [[nodiscard]] constexpr std::uint8_t operator[](std::size_t i) const { return repr_[i]; }
    [[nodiscard]] constexpr bool is_zero() const noexcept {
        return repr_[0] == 0 && repr_[1] == 0;
    }
    // 表示多项式的次数（0 或 1）。
    [[nodiscard]] constexpr std::size_t degree() const noexcept {
        return repr_[0] != 0 ? 1 : 0;
    }

    constexpr bool operator==(const Element&) const noexcept = default;

    // 加法：系数逐项异或（基域特征为 2）。
    friend constexpr Element operator+(Element a, Element b) noexcept {
        return Element(static_cast<std::uint8_t>(a.repr_[0] ^ b.repr_[0]),
                       static_cast<std::uint8_t>(a.repr_[1] ^ b.repr_[1]));
    }

    // 减法与加法相同。
    friend constexpr Element operator-(Element a, Element b) noexcept { return a + b; }

    // 乘法：两个一次多项式按 FOIL 展开成次数 <= 2 的多项式，
    // 再模 EXT_POLY 归约回次数 < 2。
    friend constexpr Element operator*(Element a, Element b) noexcept {
        const std::array<std::uint8_t, 3> out = {
            gf8::mul(a.repr_[0], b.repr_[0]),                                       // x^2 项
            gf8::add(gf8::mul(a.repr_[1], b.repr_[0]), gf8::mul(a.repr_[0], b.repr_[1])), // x 项
            gf8::mul(a.repr_[1], b.repr_[1]),                                       // 常数项
        };
        return reduce_from(out);
    }

    // 标量乘法：用基域 GF(2^8) 的元素逐系数相乘。
    friend constexpr Element operator*(Element a, std::uint8_t rhs) noexcept {
        return Element(gf8::mul(rhs, a.repr_[0]), gf8::mul(rhs, a.repr_[1]));
    }

    // 除法：乘以右操作数的逆元。
    friend constexpr Element operator/(Element a, Element b) { return a * b.inverse(); }

    // 幂运算：朴素的重复乘法（与 Rust 版一致）。约定 a^0 = 1。
    [[nodiscard]] constexpr Element exp(std::size_t n) const {
        if (n == 0) {
            return constant(1);
        }
        if (is_zero()) {
            return zero();
        }
        Element result = *this;
        for (std::size_t i = 1; i < n; ++i) {
            result = result * (*this);
        }
        return result;
    }

    // 乘法逆元。对零元抛出 std::domain_error（Rust 版 panic）。
    // 实现依赖下方 detail 命名空间的扩展欧几里得算法，故在其后定义。
    [[nodiscard]] constexpr Element inverse() const;

private:
    // 把一个次数 <= 2 的多项式归约回域元素（次数 < 2）：
    // 若 x^2 项非零，则用 EXT_POLY 做一次多项式除法并取余数。
    // 由于 EXT_POLY 首项系数为 1，余数即逐项消去：
    //   c*x^(i+j) = a*x^i * b*x^j
    static constexpr Element reduce_from(std::array<std::uint8_t, 3> x) noexcept {
        if (x[0] != 0) {
            x[1] ^= gf8::mul(EXT_POLY[1], x[0]);
            x[2] ^= gf8::mul(EXT_POLY[2], x[0]);
        }
        return Element(x[1], x[2]);
    }

    Repr repr_;
};

// 求逆所需的多项式扩展欧几里得算法。
//
// Rust 版把这些写成 Element 的私有方法 + EgcdRhs 枚举；C++ 版改为
// 自由函数放在 detail 中——嵌套结构体的成员不能使用尚不完整的外围类
// 类型，所以这些含 Element 成员的辅助结构必须定义在类外。
namespace detail {

// 扩展欧几里得的返回值：g = gcd（已知为常数），x、y 为贝祖系数。
struct Egcd {
    std::uint8_t g;
    Element x;
    Element y;
};

// 多项式除法的商和余数。
struct DivResult {
    Element quotient;
    Element remainder;
};

// 用 rhs 去除 EXT_POLY（EXT_POLY 是二次多项式，无法用 Element 表示，
// 所以单独处理）。对应 Rust 版的 Element::div_ext_by。
constexpr DivResult div_ext_by(Element rhs) {
    if (rhs.degree() == 0) {
        // 除以常数：EXT_POLY 的所有常数倍都在 0 的等价类中。
        return {Element::zero(), Element::zero()};
    }

    // 此处除数必为一次多项式；先化为首一多项式。
    const std::uint8_t leading_mul_inv = gf8::div(1, rhs[0]);
    const Element monictized = rhs * leading_mul_inv;
    std::array<std::uint8_t, 3> poly = EXT_POLY;

    // 长除法：逐项消去被除式的高次项。
    for (std::size_t i = 0; i < 2; ++i) {
        const std::uint8_t coef = poly[i];
        for (std::size_t j = 1; j < 2; ++j) {
            if (rhs[j] != 0) {
                poly[i + j] ^= gf8::mul(monictized[j], coef);
            }
        }
    }

    // 商要除回首项系数（即乘 leading_mul_inv），余数为最后剩下的常数。
    return {Element(poly[0], poly[1]) * leading_mul_inv, Element::constant(poly[2])};
}

// 多项式除法 self / rhs（均为 GF(2^8) 上次数 < 2 的多项式）。
// 对应 Rust 版的 Element::polynom_div。
constexpr DivResult polynom_div(Element self, Element rhs) {
    const std::size_t divisor_degree = rhs.degree();
    if (rhs.is_zero()) {
        throw std::domain_error("divide by 0");
    }
    if (self.degree() < divisor_degree) {
        // 除数次数更高：整个被除式都是余数。
        return {Element::zero(), self};
    }
    if (divisor_degree == 0) {
        // 除以常数：逐系数乘以常数的逆。
        const std::uint8_t invert = gf8::div(1, rhs[1]);
        return {Element(gf8::mul(invert, self[0]), gf8::mul(invert, self[1])),
                Element::zero()};
    }

    // 两者次数都为 1：把 rhs 化为首一多项式后做一步消元。
    const std::uint8_t leading_mul_inv = gf8::div(1, rhs[0]);
    const Element monic(gf8::mul(leading_mul_inv, rhs[0]), gf8::mul(leading_mul_inv, rhs[1]));

    const std::uint8_t leading_coeff = self[0];
    std::uint8_t remainder = self[1];
    if (monic[1] != 0) {
        remainder ^= gf8::mul(monic[1], self[0]);
    }

    return {Element::constant(gf8::mul(leading_mul_inv, leading_coeff)),
            Element::constant(remainder)};
}

// 两个元素之间的扩展欧几里得算法。因为 EXT_POLY 不可约，
// 最终的 GCD 必为常数。对应 Rust 版 const_egcd 的 Element 分支。
constexpr Egcd const_egcd(Element self, Element rhs) {
    if (self.is_zero()) {
        // 递归基：gcd 为 rhs 的常数项，贝祖系数 (0, 1)。
        return {rhs[1], Element::constant(0), Element::constant(1)};
    }
    const auto [q, rem] = polynom_div(rhs, self);
    const Egcd sub = const_egcd(rem, self);
    // 标准的贝祖系数回代：x' = y + q*x，y' = x。
    return {sub.g, sub.y + (q * sub.x), sub.x};
}

// 右操作数为 EXT_POLY 时的扩展欧几里得算法（第一步特殊处理）。
// 对应 Rust 版 const_egcd 的 ExtPoly 分支。
constexpr Egcd const_egcd_ext_poly(Element self) {
    const auto [q, rem] = div_ext_by(self);
    const Egcd sub = const_egcd(rem, self);
    return {sub.g, sub.y + (q * sub.x), sub.x};
}

} // namespace detail

constexpr Element Element::inverse() const {
    if (is_zero()) {
        throw std::domain_error("Cannot invert 0");
    }

    // 扩展欧几里得的第一步在这里展开：self / EXT_POLY = (0, self)，
    // 余数就是 self 本身，因此直接进入 EXT_POLY 分支。
    const detail::Egcd e = detail::const_egcd_ext_poly(*this);

    // EXT_POLY 不可约，所以 GCD 一定是非零常数。由贝祖等式
    //   EXT_POLY * y' + self * x = gcd
    // 且 EXT_POLY 的倍数等价于 0，得 self * x ≡ gcd，
    // 故 self 的逆为 x / gcd。
    if (e.g == 0) {
        throw std::domain_error("Cannot invert 0");
    }
    return e.x * gf8::div(1, e.g);
}

// 域 GF(2^16) 的类型包装，满足 rse::FieldType concept。
// 元素表示为 std::array<uint8_t, 2>（对应 Rust 版的 [u8; 2]）。
// 没有提供 mul_slice 加速，批量运算走 field.hpp 的通用逐元素循环
// ——与 Rust 版行为一致（GF(2^16) 无 SIMD 路径）。
struct Field {
    static constexpr std::size_t ORDER = 65536;
    using Elem = std::array<std::uint8_t, 2>;

    static constexpr Elem add(Elem a, Elem b) noexcept {
        return (Element(a) + Element(b)).repr();
    }
    static constexpr Elem mul(Elem a, Elem b) noexcept {
        return (Element(a) * Element(b)).repr();
    }
    static constexpr Elem div(Elem a, Elem b) { return (Element(a) / Element(b)).repr(); }
    static constexpr Elem exp(Elem a, std::size_t n) { return Element(a).exp(n).repr(); }
    static constexpr Elem zero() noexcept { return {0, 0}; }
    static constexpr Elem one() noexcept { return {0, 1}; }
    // 序号到域元素：高 8 位与低 8 位分别作为两个系数。
    static constexpr Elem nth_internal(std::size_t n) noexcept {
        return {static_cast<std::uint8_t>(n >> 8), static_cast<std::uint8_t>(n)};
    }
};

} // namespace rse::gf16
