// gf8_simd_avx512.cpp - AVX-512BW（512 位）的向量操作原语。
//
// 算法本体在 gf8_simd_kernel.hpp（各 ISA 共享）；本文件只定义
// AVX-512 的 Ops 原语并实例化内核。
// 注意 _mm512_shuffle_epi8 属于 AVX-512BW 扩展（仅 AVX-512F 不够），
// 运行时分发据此检测 "avx512bw"。
// 本文件用 -mavx512f -mavx512bw 编译；仅在运行时确认支持后才会被调用。

#include <rse/config.hpp>

#if RSE_ARCH_X86_64

#include <immintrin.h>

#include "gf8_simd_kernel.hpp"

namespace rse::gf8::detail {

namespace {

struct Avx512Ops {
    using Vec = __m512i;
    static constexpr std::size_t WIDTH = 64;

    static Vec load(const std::uint8_t* p) { return _mm512_loadu_si512(p); }
    static void store(std::uint8_t* p, Vec v) { _mm512_storeu_si512(p, v); }
    // VPSHUFB 在四个 128 位通道内各自独立查表，
    // 因此 16 字节查找表需要广播到四个通道。
    static Vec load_table(const std::uint8_t* p) {
        return _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    }
    static Vec splat(std::uint8_t c) { return _mm512_set1_epi8(static_cast<char>(c)); }
    static Vec and_(Vec a, Vec b) { return _mm512_and_si512(a, b); }
    static Vec xor_(Vec a, Vec b) { return _mm512_xor_si512(a, b); }
    // 按 64 位通道右移；跨字节窜入的位由内核的掩码清掉。
    static Vec shift_right_4(Vec v) { return _mm512_srli_epi64(v, 4); }
    static Vec lookup(Vec table, Vec idx) { return _mm512_shuffle_epi8(table, idx); }
};

} // namespace

SimdKernels avx512_kernels() noexcept { return make_kernels<Avx512Ops>("avx512"); }

} // namespace rse::gf8::detail

#endif // RSE_ARCH_X86_64
