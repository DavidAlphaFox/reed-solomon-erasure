// galois_8.hpp - 有限域 GF(2^8) 的算术实现，生成多项式为 0x11D（即 29 + x^8）。
//
// 移植自 Rust crate 的 src/galois_8.rs。
// Rust 版在 build.rs 中于构建期生成查找表再 include 进源码；
// C++23 版直接用 consteval 函数在编译期生成同样的表，无需代码生成步骤。
//
// 域的表示：以 log/exp 表为基础——
//   LOG_TABLE[a]  = log_g(a)（a 以生成元 g=2 为底的离散对数）
//   EXP_TABLE[n]  = g^n（表长 510，复制一份以省去取模）
//   MUL_TABLE[a][b] = a * b（完整的 256x256 = 64KiB 乘法表）
//   MUL_TABLE_LOW / HIGH = SIMD 内核使用的"低/高半字节"拆分表
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

namespace rse::gf8 {

inline constexpr std::size_t FIELD_SIZE = 256;                  // 域的阶 2^8
inline constexpr std::size_t EXP_TABLE_SIZE = FIELD_SIZE * 2 - 2; // exp 表双倍长度，免取模
inline constexpr unsigned GENERATING_POLYNOMIAL = 29;            // x^8+x^4+x^3+x^2+1 的低 8 位

namespace detail {

// 生成离散对数表：从 1 开始反复乘以生成元（左移一位），
// 溢出 8 位时按生成多项式归约。对应 build.rs 的 gen_log_table。
consteval std::array<std::uint8_t, FIELD_SIZE> gen_log_table(unsigned polynomial) {
    std::array<std::uint8_t, FIELD_SIZE> result{};
    std::size_t b = 1;
    for (std::size_t log = 0; log < FIELD_SIZE - 1; ++log) {
        result[b] = static_cast<std::uint8_t>(log);
        b <<= 1;
        if (FIELD_SIZE <= b) {
            b = (b - FIELD_SIZE) ^ polynomial;
        }
    }
    return result;
}

// 由对数表反推出幂表。表长 510：前 255 项是 g^0..g^254，
// 后面再复制一份，使 EXP_TABLE[log_a + log_b] 不需要对 255 取模。
// 对应 build.rs 的 gen_exp_table。
consteval std::array<std::uint8_t, EXP_TABLE_SIZE>
gen_exp_table(const std::array<std::uint8_t, FIELD_SIZE>& log_table) {
    std::array<std::uint8_t, EXP_TABLE_SIZE> result{};
    for (std::size_t i = 1; i < FIELD_SIZE; ++i) {
        const auto log = log_table[i];
        result[log] = static_cast<std::uint8_t>(i);
        result[log + FIELD_SIZE - 1] = static_cast<std::uint8_t>(i);
    }
    return result;
}

} // namespace detail

inline constexpr auto LOG_TABLE = detail::gen_log_table(GENERATING_POLYNOMIAL);
inline constexpr auto EXP_TABLE = detail::gen_exp_table(LOG_TABLE);

namespace detail {

// 表生成期间使用的乘法：a*b = g^(log a + log b)，0 与任何数相乘为 0。
constexpr std::uint8_t multiply(std::uint8_t a, std::uint8_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return EXP_TABLE[static_cast<std::size_t>(LOG_TABLE[a]) + LOG_TABLE[b]];
}

// 生成完整的 256x256 乘法表。运行期乘法即查一次表。
// 注意：在一次常量求值中要做 65536 次乘法，超过 clang/MSVC 默认的
// constexpr 求值步数上限，因此构建系统会传入
// -fconstexpr-steps / /constexpr:steps 提高限制（见 CMakeLists.txt）。
consteval std::array<std::array<std::uint8_t, FIELD_SIZE>, FIELD_SIZE> gen_mul_table() {
    std::array<std::array<std::uint8_t, FIELD_SIZE>, FIELD_SIZE> result{};
    for (std::size_t a = 0; a < FIELD_SIZE; ++a) {
        for (std::size_t b = 0; b < FIELD_SIZE; ++b) {
            result[a][b] = multiply(static_cast<std::uint8_t>(a), static_cast<std::uint8_t>(b));
        }
    }
    return result;
}

// SIMD 内核使用的"半字节拆分"表：对每个乘数 a，
//   low[a][x]  = a * x          （x 为低 4 位，x < 16）
//   high[a][x] = a * (x << 4)   （x 为高 4 位）
// 由于 GF(2^n) 中乘法对加法（异或）满足分配律：
//   a * b = a * (b_low ^ b_high) = (a*b_low) ^ (a*b_high)
// 因此可用两次 16 项查表（PSHUFB/TBL 指令）合成一次完整乘法。
// 对应 build.rs 的 gen_mul_table_half。
struct HalfTables {
    std::array<std::array<std::uint8_t, 16>, FIELD_SIZE> low;
    std::array<std::array<std::uint8_t, 16>, FIELD_SIZE> high;
};

consteval HalfTables gen_mul_table_half() {
    HalfTables t{};
    for (std::size_t a = 0; a < FIELD_SIZE; ++a) {
        for (std::size_t b = 0; b < FIELD_SIZE; ++b) {
            const auto result =
                multiply(static_cast<std::uint8_t>(a), static_cast<std::uint8_t>(b));
            if ((b & 0x0F) == b) {
                t.low[a][b] = result;
            }
            if ((b & 0xF0) == b) {
                t.high[a][b >> 4] = result;
            }
        }
    }
    return t;
}

} // namespace detail

inline constexpr auto MUL_TABLE = detail::gen_mul_table();
inline constexpr auto MUL_TABLE_HALF = detail::gen_mul_table_half();
inline constexpr auto& MUL_TABLE_LOW = MUL_TABLE_HALF.low;
inline constexpr auto& MUL_TABLE_HIGH = MUL_TABLE_HALF.high;

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
    // log(a/b) = log a - log b（在模 255 意义下）
    int log_result = static_cast<int>(LOG_TABLE[a]) - static_cast<int>(LOG_TABLE[b]);
    if (log_result < 0) {
        log_result += 255;
    }
    return EXP_TABLE[static_cast<std::size_t>(log_result)];
}

// 幂运算 a^n：log(a^n) = n * log a（模 255）。
// 约定 a^0 = 1（包括 0^0 = 1，与 Rust 版一致）。
[[nodiscard]] constexpr std::uint8_t exp(std::uint8_t a, std::size_t n) noexcept {
    if (n == 0) {
        return 1;
    }
    if (a == 0) {
        return 0;
    }
    std::size_t log_result = static_cast<std::size_t>(LOG_TABLE[a]) * n;
    while (255 <= log_result) {
        log_result -= 255;
    }
    return EXP_TABLE[log_result];
}

// --- 批量切片运算（实现在 src/galois_8.cpp，带运行时 SIMD 分发）---

// out[i] = c * input[i]。
// 主体由运行时选出的 SIMD 内核处理，尾部由标量查表补齐；
// 两个 span 长度不一致时抛出 std::invalid_argument（Rust 版 assert）。
void mul_slice(std::uint8_t c, std::span<const std::uint8_t> input, std::span<std::uint8_t> out);

// out[i] ^= c * input[i]（乘后异或累加），加速方式同 mul_slice。
void mul_slice_xor(std::uint8_t c, std::span<const std::uint8_t> input,
                   std::span<std::uint8_t> out);

// out[i] ^= input[i]（纯异或，测试中使用）。
void slice_xor(std::span<const std::uint8_t> input, std::span<std::uint8_t> out);

// 返回运行时实际选中的 SIMD 后端名称：
// "avx512" / "avx2" / "ssse3" / "neon" / "scalar"。
// 供诊断与测试使用。
[[nodiscard]] std::string_view simd_backend() noexcept;

// 域 GF(2^8) 的类型包装，满足 rse::FieldType concept。
// 对应 Rust 版的 galois_8::Field（实现 Field trait）。
struct Field {
    static constexpr std::size_t ORDER = 256;
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
