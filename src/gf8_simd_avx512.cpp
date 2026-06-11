// gf8_simd_avx512.cpp - AVX-512BW（512 位）GF(2^8) 乘法内核。
//
// 移植自 simd_c/reedsolomon.c 的 AVX512 路径（版权归 Nicolas Trangez 与
// Klaus Post 所有，MIT 许可，见仓库根目录 LICENSE）。
//
// 算法与 SSSE3/AVX2 版相同（半字节拆分 + 字节 shuffle 查表，
// 详见 gf8_simd_ssse3.cpp 的注释），向量宽度为 64 字节。
// 注意 _mm512_shuffle_epi8 属于 AVX-512BW 扩展（仅 AVX-512F 不够），
// 运行时分发据此检测 "avx512bw"。
//
// 本文件用 -mavx512f -mavx512bw 编译；仅在运行时确认支持后才会被调用。

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "galois_simd.hpp"

namespace rse::gf8::detail {

namespace {

// XOR 含义同 SSSE3 版：false 覆盖写，true 异或累加。
template <bool XOR>
std::size_t gal_mul_avx512(const std::uint8_t* low, const std::uint8_t* high,
                           const std::uint8_t* in, std::uint8_t* out, std::size_t len) {
    const __m512i low_mask = _mm512_set1_epi8(0x0f);
    // 把 16 字节查找表广播到 512 位向量的四个 128 位通道
    // （VPSHUFB 在每个通道内独立查表）。
    const __m512i low_vector = _mm512_broadcast_i32x4(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(low)));
    const __m512i high_vector = _mm512_broadcast_i32x4(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(high)));

    std::size_t done = 0;
    for (std::size_t x = 0; x < len / 64; ++x) {
        const __m512i in_x = _mm512_loadu_si512(in + done);
        const __m512i low_input = _mm512_and_si512(in_x, low_mask);
        const __m512i high_input = _mm512_and_si512(_mm512_srli_epi64(in_x, 4), low_mask);
        const __m512i mul_low = _mm512_shuffle_epi8(low_vector, low_input);
        const __m512i mul_high = _mm512_shuffle_epi8(high_vector, high_input);
        __m512i result = _mm512_xor_si512(mul_low, mul_high);
        if constexpr (XOR) {
            const __m512i old = _mm512_loadu_si512(out + done);
            result = _mm512_xor_si512(result, old);
        }
        _mm512_storeu_si512(out + done, result);
        done += 64;
    }
    return done; // 尾部（< 64 字节）由调用方标量处理
}

} // namespace

SimdKernels avx512_kernels() noexcept {
    return {&gal_mul_avx512<false>, &gal_mul_avx512<true>, "avx512"};
}

} // namespace rse::gf8::detail

#endif // x86_64
