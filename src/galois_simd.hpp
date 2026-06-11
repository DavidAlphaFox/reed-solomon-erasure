// galois_simd.hpp - GF(2^8) SIMD 内核的内部声明（不对外安装）。
//
// 每个内核对一段字节切片做 GF(2^8) 常数乘法，使用"半字节拆分 + 字节
// shuffle"技巧（详见 galois_8.hpp 中 MUL_TABLE_LOW/HIGH 的注释）：
//   low / high  : 当前乘数对应的两张 16 字节查找表
//   mul         : out[i] = c * in[i]（覆盖写）
//   mul_xor     : out[i] ^= c * in[i]（异或累加）
// 内核只处理整向量宽度的部分并返回已处理的字节数；
// 剩余的尾部由调用方（galois_8.cpp）用标量查表补齐。
//
// 各 ISA 的内核分别放在单独的翻译单元中、用对应的 -m 编译选项编译，
// 运行时由 active_kernels() 按 CPU 能力选择，因此库本身无需
// -march 即可在老 CPU 上安全运行。
#pragma once

#include <cstddef>
#include <cstdint>

namespace rse::gf8::detail {

// 内核函数指针类型：返回值为已处理的字节数（向量宽度的整数倍）。
using GalMulFn = std::size_t (*)(const std::uint8_t* low, const std::uint8_t* high,
                                 const std::uint8_t* in, std::uint8_t* out, std::size_t len);

// 一组内核（覆盖写 + 异或累加）及其后端名称。
struct SimdKernels {
    GalMulFn mul;
    GalMulFn mul_xor;
    const char* name; // "avx512" / "avx2" / "ssse3" / "neon" / "scalar"
};

#if defined(__x86_64__) || defined(_M_X64)
SimdKernels ssse3_kernels() noexcept;  // 128 位，gf8_simd_ssse3.cpp
SimdKernels avx2_kernels() noexcept;   // 256 位，gf8_simd_avx2.cpp
SimdKernels avx512_kernels() noexcept; // 512 位，gf8_simd_avx512.cpp
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
SimdKernels neon_kernels() noexcept;   // 128 位，gf8_simd_neon.cpp
#endif

// 返回为当前 CPU 选出的内核（首次调用时检测一次，线程安全）。
const SimdKernels& active_kernels() noexcept;

} // namespace rse::gf8::detail
