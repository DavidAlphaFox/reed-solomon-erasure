// gf8_simd_ssse3.cpp - SSSE3（128 位）的向量操作原语。
//
// 算法本体在 gf8_simd_kernel.hpp（各 ISA 共享）；本文件只定义
// SSSE3 的 Ops 原语并实例化内核。
// 本文件用 -mssse3 编译；仅在运行时 CPU 检测确认支持后才会被调用。

#include <rse/config.hpp>

#if RSE_ARCH_X86_64

#include <tmmintrin.h>

#include "gf8_simd_kernel.hpp"

namespace rse::gf8::detail {

namespace {

struct Ssse3Ops {
    using Vec = __m128i;
    static constexpr std::size_t WIDTH = 16;

    static Vec load(const std::uint8_t* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    static void store(std::uint8_t* p, Vec v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    }
    // 128 位向量恰好装下一张 16 字节查找表，无需广播。
    static Vec load_table(const std::uint8_t* p) { return load(p); }
    static Vec splat(std::uint8_t c) { return _mm_set1_epi8(static_cast<char>(c)); }
    static Vec and_(Vec a, Vec b) { return _mm_and_si128(a, b); }
    static Vec xor_(Vec a, Vec b) { return _mm_xor_si128(a, b); }
    // PSRLQ 按 64 位通道右移；跨字节窜入的位由内核的掩码清掉。
    static Vec shift_right_4(Vec v) { return _mm_srli_epi64(v, 4); }
    static Vec lookup(Vec table, Vec idx) { return _mm_shuffle_epi8(table, idx); }
};

} // namespace

SimdKernels ssse3_kernels() noexcept { return make_kernels<Ssse3Ops>("ssse3"); }

} // namespace rse::gf8::detail

#endif // RSE_ARCH_X86_64
