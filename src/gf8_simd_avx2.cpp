// gf8_simd_avx2.cpp - AVX2（256 位）GF(2^8) 乘法内核。
//
// 移植自 simd_c/reedsolomon.c 的 AVX2 路径（版权归 Nicolas Trangez 与
// Klaus Post 所有，MIT 许可，见仓库根目录 LICENSE）。
//
// 算法与 SSSE3 版完全相同（半字节拆分 + 字节 shuffle 查表，
// 详见 gf8_simd_ssse3.cpp 的注释），只是向量宽度翻倍：
// VPSHUFB 在 256 位向量的两个 128 位通道内各自独立查表，
// 因此 16 字节的查找表需要广播到两个通道（broadcastsi128）。
//
// 本文件用 -mavx2 编译；仅在运行时 CPU 检测确认支持后才会被调用。

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "galois_simd.hpp"

namespace rse::gf8::detail {

namespace {

// XOR 含义同 SSSE3 版：false 覆盖写，true 异或累加。
template <bool XOR>
std::size_t gal_mul_avx2(const std::uint8_t* low, const std::uint8_t* high,
                         const std::uint8_t* in, std::uint8_t* out, std::size_t len) {
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    // 把 16 字节查找表广播到 256 位向量的两个通道。
    const __m256i low_vector = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(low)));
    const __m256i high_vector = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(high)));

    std::size_t done = 0;
    for (std::size_t x = 0; x < len / 32; ++x) {
        const __m256i in_x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + done));
        const __m256i low_input = _mm256_and_si256(in_x, low_mask);
        const __m256i high_input = _mm256_and_si256(_mm256_srli_epi64(in_x, 4), low_mask);
        const __m256i mul_low = _mm256_shuffle_epi8(low_vector, low_input);
        const __m256i mul_high = _mm256_shuffle_epi8(high_vector, high_input);
        __m256i result = _mm256_xor_si256(mul_low, mul_high);
        if constexpr (XOR) {
            const __m256i old =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + done));
            result = _mm256_xor_si256(result, old);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + done), result);
        done += 32;
    }
    return done; // 尾部（< 32 字节）由调用方标量处理
}

} // namespace

SimdKernels avx2_kernels() noexcept {
    return {&gal_mul_avx2<false>, &gal_mul_avx2<true>, "avx2"};
}

} // namespace rse::gf8::detail

#endif // x86_64
