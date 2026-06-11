// gf8_simd_ssse3.cpp - SSSE3（128 位）GF(2^8) 乘法内核。
//
// 移植自 simd_c/reedsolomon.c 的 SSE2/SSSE3 路径，该文件版权归
// Nicolas Trangez (c) 2015, 2016 与 Klaus Post (c) 2015 所有（MIT 许可，
// 见仓库根目录 LICENSE）。算法出自 "Screaming Fast Galois Field
// Arithmetic Using Intel SIMD Instructions" (Plank et al.)。
//
// 算法（每 16 字节一组）：
//   1. 把每个输入字节 b 拆成高低两个半字节：lo = b & 0x0F，hi = b >> 4；
//   2. 用 PSHUFB 以 lo / hi 为下标分别查 low / high 两张 16 项表，
//      得到 c*lo 和 c*(hi<<4)；
//   3. 两者异或即 c*b（GF(2^8) 乘法对异或满足分配律）。
//
// 本文件用 -mssse3 编译；仅在运行时 CPU 检测确认支持后才会被调用。

#if defined(__x86_64__) || defined(_M_X64)

#include <tmmintrin.h>

#include "galois_simd.hpp"

namespace rse::gf8::detail {

namespace {

// XOR 为 false 时覆盖写 out，为 true 时与 out 原值异或（乘加）。
// 两个变体共享同一份模板，编译期展开，无运行时分支。
template <bool XOR>
std::size_t gal_mul_ssse3(const std::uint8_t* low, const std::uint8_t* high,
                          const std::uint8_t* in, std::uint8_t* out, std::size_t len) {
    const __m128i low_mask = _mm_set1_epi8(0x0f);
    // 两张 16 字节查找表各装入一个向量寄存器。
    const __m128i low_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(low));
    const __m128i high_vector = _mm_loadu_si128(reinterpret_cast<const __m128i*>(high));

    std::size_t done = 0;
    for (std::size_t x = 0; x < len / 16; ++x) {
        const __m128i in_x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + done));
        // 拆半字节。注意 PSRLQ 按 64 位通道右移，跨字节窜入的位
        // 会被随后的 & 0x0F 掩掉，结果等价于按字节右移 4。
        const __m128i low_input = _mm_and_si128(in_x, low_mask);
        const __m128i high_input = _mm_and_si128(_mm_srli_epi64(in_x, 4), low_mask);
        // PSHUFB 并行查表。
        const __m128i mul_low = _mm_shuffle_epi8(low_vector, low_input);
        const __m128i mul_high = _mm_shuffle_epi8(high_vector, high_input);
        __m128i result = _mm_xor_si128(mul_low, mul_high);
        if constexpr (XOR) {
            const __m128i old = _mm_loadu_si128(reinterpret_cast<const __m128i*>(out + done));
            result = _mm_xor_si128(result, old);
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + done), result);
        done += 16;
    }
    return done; // 尾部（< 16 字节）由调用方标量处理
}

} // namespace

SimdKernels ssse3_kernels() noexcept {
    return {&gal_mul_ssse3<false>, &gal_mul_ssse3<true>, "ssse3"};
}

} // namespace rse::gf8::detail

#endif // x86_64
