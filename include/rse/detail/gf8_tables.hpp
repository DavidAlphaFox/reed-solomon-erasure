// gf8_tables.hpp - GF(2^8) 查找表的编译期生成（内部实现细节）。
//
// 从 galois_8.hpp 中拆出，使公共头只暴露域运算接口，表生成逻辑
// 内聚在此处。Rust 版在 build.rs 中于构建期生成这些表再 include
// 进源码；C++23 版用 consteval 在编译期完成，无需代码生成步骤。
//
// 表的体系：
//   LOG_TABLE[a]    = log_g(a)（a 以生成元 g=2 为底的离散对数）
//   EXP_TABLE[n]    = g^n（表长 510，复制一份以省去取模）
//   MUL_TABLE[a][b] = a * b（完整的 256x256 = 64KiB 乘法表）
//   MUL_TABLE_LOW / HIGH = SIMD 内核使用的"低/高半字节"拆分表
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace rse::gf8 {

inline constexpr std::size_t FIELD_SIZE = 256;       // 域的阶 2^8
inline constexpr std::size_t GROUP_ORDER = FIELD_SIZE - 1; // 乘法群的阶（非零元个数）
inline constexpr std::size_t EXP_TABLE_SIZE = FIELD_SIZE * 2 - 2; // exp 表双倍长度，免取模
inline constexpr unsigned GENERATING_POLYNOMIAL = 29; // x^8+x^4+x^3+x^2+1 的低 8 位

namespace detail {

// 生成离散对数表：从 1 开始反复乘以生成元（左移一位），
// 溢出 8 位时按生成多项式归约。对应 Rust build.rs 的 gen_log_table。
consteval std::array<std::uint8_t, FIELD_SIZE> gen_log_table(unsigned polynomial) {
    std::array<std::uint8_t, FIELD_SIZE> result{};
    std::size_t b = 1;
    for (std::size_t log = 0; log < GROUP_ORDER; ++log) {
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
consteval std::array<std::uint8_t, EXP_TABLE_SIZE>
gen_exp_table(const std::array<std::uint8_t, FIELD_SIZE>& log_table) {
    std::array<std::uint8_t, EXP_TABLE_SIZE> result{};
    for (std::size_t i = 1; i < FIELD_SIZE; ++i) {
        const auto log = log_table[i];
        result[log] = static_cast<std::uint8_t>(i);
        result[log + GROUP_ORDER] = static_cast<std::uint8_t>(i);
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

} // namespace rse::gf8
