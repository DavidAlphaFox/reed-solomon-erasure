// galois_8.cpp - GF(2^8) 批量切片运算的实现，含运行时 SIMD 分发。
//
// 移植自 Rust crate 的 src/galois_8.rs（标量路径）以及
// simd_c/reedsolomon.c（SIMD 路径的调度逻辑）。
//
// 工作划分：
//   - SIMD 内核（x86-64 上的 SSSE3/AVX2/AVX-512BW、AArch64 上的 NEON）
//     处理切片中向量宽度整数倍的主体部分；
//   - 标量查表循环补齐尾部，同时也是无 SIMD 平台的完整后备实现。
// 与 Rust 版的差异：Rust 版在编译期通过 feature/target 选择实现，
// 这里改为运行期 CPU 检测，一个二进制可适配所有机器。

#include "rse/galois_8.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "galois_simd.hpp"

namespace rse::gf8 {

namespace {

// 标量实现：逐字节查 64KiB 乘法表。XOR 为 false 时覆盖写，为 true 时
// 异或累加——与 SIMD 内核同样的编译期双形态，消除成对的重复函数。
// （对应 Rust 版的 mul_slice_pure_rust / mul_slice_xor_pure_rust。）
template <bool XOR>
void mul_slice_scalar(std::uint8_t c, std::span<const std::uint8_t> input,
                      std::span<std::uint8_t> out) {
    const auto& mt = MUL_TABLE[c];
    for (std::size_t i = 0; i < input.size(); ++i) {
        if constexpr (XOR) {
            out[i] ^= mt[input[i]];
        } else {
            out[i] = mt[input[i]];
        }
    }
}

// 输入输出长度必须一致（Rust 版此处为 assert_eq!，即 panic）。
void check_same_length(std::span<const std::uint8_t> input, std::span<std::uint8_t> out) {
    if (input.size() != out.size()) {
        throw std::invalid_argument("gf8: input/output length mismatch");
    }
}

// mul_slice / mul_slice_xor 的公共骨架：
// 长度检查 -> SIMD 内核处理主体 -> 标量补齐尾部。
template <bool XOR>
void mul_slice_impl(std::uint8_t c, std::span<const std::uint8_t> input,
                    std::span<std::uint8_t> out) {
    check_same_length(input, out);
    if (input.empty()) {
        return;
    }

    const auto& kernels = detail::active_kernels();
    const auto kernel = XOR ? kernels.mul_xor : kernels.mul;
    const std::size_t done =
        kernel(MUL_TABLE_LOW[c].data(), MUL_TABLE_HIGH[c].data(), input.data(), out.data(),
               input.size());

    mul_slice_scalar<XOR>(c, input.subspan(done), out.subspan(done));
}

} // namespace

namespace detail {

namespace {

// 无 SIMD 时的占位内核：声称处理了 0 字节，让标量尾部循环干全部活。
std::size_t scalar_noop(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
                        std::uint8_t*, std::size_t) {
    return 0;
}

// 运行时 CPU 能力检测，按"越宽越好"的顺序选择内核。
// 只在进程内执行一次（见 active_kernels 中的静态局部变量）。
SimdKernels detect_kernels() noexcept {
#if RSE_ARCH_X86_64
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    // _mm512_shuffle_epi8 需要 AVX-512BW（仅 AVX-512F 不够）。
    if (__builtin_cpu_supports("avx512bw")) {
        return avx512_kernels();
    }
    if (__builtin_cpu_supports("avx2")) {
        return avx2_kernels();
    }
    if (__builtin_cpu_supports("ssse3")) {
        return ssse3_kernels();
    }
#else
    // MSVC 等没有 __builtin_cpu_supports 的编译器：SSSE3 在 x86-64
    // 上早已普及，保守起见只启用到 SSSE3。
    return ssse3_kernels();
#endif
#elif RSE_ARCH_AARCH64
    // NEON 是 AArch64 的基线特性，无需检测。
    return neon_kernels();
#endif
    return {&scalar_noop, &scalar_noop, "scalar"};
}

} // namespace

const SimdKernels& active_kernels() noexcept {
    // C++11 静态局部变量初始化自带线程安全保证。
    static const SimdKernels kernels = detect_kernels();
    return kernels;
}

} // namespace detail

std::string_view simd_backend() noexcept { return detail::active_kernels().name; }

void mul_slice(std::uint8_t c, std::span<const std::uint8_t> input,
               std::span<std::uint8_t> out) {
    mul_slice_impl<false>(c, input, out);
}

void mul_slice_xor(std::uint8_t c, std::span<const std::uint8_t> input,
                   std::span<std::uint8_t> out) {
    mul_slice_impl<true>(c, input, out);
}

void slice_xor(std::span<const std::uint8_t> input, std::span<std::uint8_t> out) {
    check_same_length(input, out);
    for (std::size_t i = 0; i < input.size(); ++i) {
        out[i] ^= input[i];
    }
}

} // namespace rse::gf8
