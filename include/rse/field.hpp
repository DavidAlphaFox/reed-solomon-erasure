// field.hpp - 编解码器所泛化的"有限域"概念（concept）定义。
//
// 移植自 Rust crate 的 src/lib.rs 中的 `Field` trait。
// Rust 用 trait 约束泛型参数；C++23 用 concept 实现同样的角色：
// ReedSolomon<F>、Matrix<F> 等模板都要求 F 满足 FieldType。
#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace rse {

// 进行编码运算所基于的有限域。
//
// 一个满足该 concept 的类型 F 需要提供：
//   - F::Elem  : 域元素的表示类型（GF(2^8) 为 uint8_t，
//                GF(2^16) 为 std::array<uint8_t, 2>）；
//   - F::ORDER : 域的阶（元素个数），同时是分片总数的上限；
//   - 域算术  : add / mul / div / exp，以及加法单位元 zero()、
//                乘法单位元 one()；
//   - nth_internal(n) : 把序号 n 映射为第 n 个域元素（用于构造
//                Vandermonde 矩阵，映射方式任意但必须一一对应）。
//
// 可选地，F 还可以提供 mul_slice / mul_slice_add 批量运算的加速实现
// （GF(2^8) 提供了 SIMD 版本）；没有提供时退化为下面的逐元素循环。
template <typename F>
concept FieldType = requires(typename F::Elem e, std::size_t n) {
    requires std::semiregular<typename F::Elem>;
    requires std::equality_comparable<typename F::Elem>;
    { F::ORDER } -> std::convertible_to<std::size_t>;
    { F::add(e, e) } -> std::same_as<typename F::Elem>;
    { F::mul(e, e) } -> std::same_as<typename F::Elem>;
    { F::div(e, e) } -> std::same_as<typename F::Elem>;
    { F::exp(e, n) } -> std::same_as<typename F::Elem>;
    { F::zero() } -> std::same_as<typename F::Elem>;
    { F::one() } -> std::same_as<typename F::Elem>;
    { F::nth_internal(n) } -> std::same_as<typename F::Elem>;
};

namespace field {

// 返回域中第 n 个元素。
// n >= F::ORDER 时抛出 std::out_of_range（Rust 版此处 panic）。
template <FieldType F>
[[nodiscard]] constexpr typename F::Elem nth(std::size_t n) {
    if (n >= F::ORDER) {
        throw std::out_of_range("field element index out of bounds");
    }
    return F::nth_internal(n);
}

// 批量乘法：out[i] = elem * input[i]。
//
// 如果域类型 F 自带 mul_slice 加速实现（编译期用 requires 探测），
// 则转发过去；否则使用通用的逐元素循环（对应 Rust 版 Field trait
// 的默认实现）。长度不一致时抛出 std::invalid_argument。
template <FieldType F>
void mul_slice(typename F::Elem elem, std::span<const typename F::Elem> input,
               std::span<typename F::Elem> out) {
    if constexpr (requires { F::mul_slice(elem, input, out); }) {
        F::mul_slice(elem, input, out);
    } else {
        if (input.size() != out.size()) {
            throw std::invalid_argument("mul_slice: input/output length mismatch");
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            out[i] = F::mul(elem, input[i]);
        }
    }
}

// 批量乘加：out[i] = out[i] + elem * input[i]。
// 在 GF(2^n) 中加法即异或，所以这等价于"乘后异或累加"。
// 分发逻辑与 mul_slice 相同。
template <FieldType F>
void mul_slice_add(typename F::Elem elem, std::span<const typename F::Elem> input,
                   std::span<typename F::Elem> out) {
    if constexpr (requires { F::mul_slice_add(elem, input, out); }) {
        F::mul_slice_add(elem, input, out);
    } else {
        if (input.size() != out.size()) {
            throw std::invalid_argument("mul_slice_add: input/output length mismatch");
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            out[i] = F::add(out[i], F::mul(elem, input[i]));
        }
    }
}

} // namespace field

} // namespace rse
