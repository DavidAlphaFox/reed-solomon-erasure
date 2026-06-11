// galois_8.hpp - 有限域 GF(2^8) 的算术接口，生成多项式为 0x11D（即 29 + x^8）。
//
// 移植自 Rust crate 的 src/galois_8.rs。
// 查找表的编译期生成在 detail/gf8_tables.hpp（实现细节）；
// 本头文件只暴露域运算与批量切片运算的公共接口。
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

#include "config.hpp"
#include "detail/gf8_tables.hpp"

namespace rse::gf8 {

// --- 单元素运算 ---

// 加法：GF(2^n) 中即按位异或。
[[nodiscard]] constexpr std::uint8_t add(std::uint8_t a, std::uint8_t b) noexcept { return a ^ b; }

// 减法：特征为 2 的域中减法与加法相同。
[[nodiscard]] constexpr std::uint8_t sub(std::uint8_t a, std::uint8_t b) noexcept { return a ^ b; }

// 乘法：直接查 64KiB 乘法表。
[[nodiscard]] constexpr std::uint8_t mul(std::uint8_t a, std::uint8_t b) noexcept {
    return MUL_TABLE[a][b];
}

// 除法 a / b：通过对数相减实现。
// b 为 0 时抛出 std::domain_error（Rust 版此处 panic）。
[[nodiscard]] constexpr std::uint8_t div(std::uint8_t a, std::uint8_t b) {
    if (a == 0) {
        return 0;
    }
    if (b == 0) {
        throw std::domain_error("Divisor is 0");
    }
    // log(a/b) = log a - log b（在模乘法群阶的意义下）
    int log_result = static_cast<int>(LOG_TABLE[a]) - static_cast<int>(LOG_TABLE[b]);
    if (log_result < 0) {
        log_result += static_cast<int>(GROUP_ORDER);
    }
    return EXP_TABLE[static_cast<std::size_t>(log_result)];
}

// 幂运算 a^n：log(a^n) = n * log a（模乘法群阶）。
// 约定 a^0 = 1（包括 0^0 = 1，与 Rust 版一致）。
[[nodiscard]] constexpr std::uint8_t exp(std::uint8_t a, std::size_t n) noexcept {
    if (n == 0) {
        return 1;
    }
    if (a == 0) {
        return 0;
    }
    std::size_t log_result = static_cast<std::size_t>(LOG_TABLE[a]) * n;
    while (GROUP_ORDER <= log_result) {
        log_result -= GROUP_ORDER;
    }
    return EXP_TABLE[log_result];
}

// --- 批量切片运算（实现在 src/galois_8.cpp，带运行时 SIMD 分发）---

// out[i] = c * input[i]。
// 主体由运行时选出的 SIMD 内核处理，尾部由标量查表补齐；
// 两个 span 长度不一致时抛出 std::invalid_argument（Rust 版 assert）。
RSE_API void mul_slice(std::uint8_t c, std::span<const std::uint8_t> input,
                       std::span<std::uint8_t> out);

// out[i] ^= c * input[i]（乘后异或累加），加速方式同 mul_slice。
RSE_API void mul_slice_xor(std::uint8_t c, std::span<const std::uint8_t> input,
                           std::span<std::uint8_t> out);

// out[i] ^= input[i]（纯异或，测试中使用）。
RSE_API void slice_xor(std::span<const std::uint8_t> input, std::span<std::uint8_t> out);

// 返回运行时实际选中的 SIMD 后端名称：
// "avx512" / "avx2" / "ssse3" / "neon" / "scalar"。
// 供诊断与测试使用。
[[nodiscard]] RSE_API std::string_view simd_backend() noexcept;

// 域 GF(2^8) 的类型包装，满足 rse::FieldType concept。
// 对应 Rust 版的 galois_8::Field（实现 Field trait）。
struct Field {
    static constexpr std::size_t ORDER = FIELD_SIZE;
    using Elem = std::uint8_t;

    static constexpr Elem add(Elem a, Elem b) noexcept { return gf8::add(a, b); }
    static constexpr Elem mul(Elem a, Elem b) noexcept { return gf8::mul(a, b); }
    static constexpr Elem div(Elem a, Elem b) { return gf8::div(a, b); }
    static constexpr Elem exp(Elem a, std::size_t n) noexcept { return gf8::exp(a, n); }
    static constexpr Elem zero() noexcept { return 0; }
    static constexpr Elem one() noexcept { return 1; }
    // 序号到域元素的映射：GF(2^8) 直接取低 8 位即可。
    static constexpr Elem nth_internal(std::size_t n) noexcept {
        return static_cast<Elem>(n);
    }

    // 提供批量运算的加速实现；field::mul_slice<F> 会探测到并转发到这里。
    static void mul_slice(Elem c, std::span<const Elem> input, std::span<Elem> out) {
        gf8::mul_slice(c, input, out);
    }
    static void mul_slice_add(Elem c, std::span<const Elem> input, std::span<Elem> out) {
        gf8::mul_slice_xor(c, input, out);
    }
};

} // namespace rse::gf8
