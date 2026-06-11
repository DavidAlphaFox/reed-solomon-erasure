// gf8_simd_avx2.cpp - AVX2（256 位）的向量操作原语。
//
// 算法本体在 gf8_simd_kernel.hpp（各 ISA 共享）；本文件只定义
// AVX2 的 Ops 原语并实例化内核。
// 本文件用 -mavx2 编译；仅在运行时 CPU 检测确认支持后才会被调用。

#include <rse/config.hpp>

#if RSE_ARCH_X86_64

#include <immintrin.h>

#include "gf8_simd_kernel.hpp"

namespace rse::gf8::detail {

namespace {

struct Avx2Ops {
    using Vec = __m256i;
    static constexpr std::size_t WIDTH = 32;

    static Vec load(const std::uint8_t* p) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static void store(std::uint8_t* p, Vec v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    // VPSHUFB 在两个 128 位通道内各自独立查表，
    // 因此 16 字节查找表需要广播到两个通道。
    static Vec load_table(const std::uint8_t* p) {
        return _mm256_broadcastsi128_si256(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    }
    static Vec splat(std::uint8_t c) { return _mm256_set1_epi8(static_cast<char>(c)); }
    static Vec and_(Vec a, Vec b) { return _mm256_and_si256(a, b); }
    static Vec xor_(Vec a, Vec b) { return _mm256_xor_si256(a, b); }
    // 按 64 位通道右移；跨字节窜入的位由内核的掩码清掉。
    static Vec shift_right_4(Vec v) { return _mm256_srli_epi64(v, 4); }
    static Vec lookup(Vec table, Vec idx) { return _mm256_shuffle_epi8(table, idx); }
};

} // namespace

SimdKernels avx2_kernels() noexcept { return make_kernels<Avx2Ops>("avx2"); }

} // namespace rse::gf8::detail

#endif // RSE_ARCH_X86_64
