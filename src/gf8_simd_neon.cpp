// gf8_simd_neon.cpp - ARM NEON（128 位）的向量操作原语。
//
// 算法本体在 gf8_simd_kernel.hpp（各 ISA 共享）；本文件只定义
// NEON 的 Ops 原语并实例化内核。查表指令用 AArch64 的 TBL
// （vqtbl1q_u8，与 x86 PSHUFB 等价的 16 字节并行查表）。
// NEON 是 AArch64 的基线特性，无需特殊指令集开关与运行时检测。
//
// 移植注意事项：某些工具链（如 OpenBSD clang）在 AArch64 上默认
// +strict-align，会把非对齐向量加载 vld1q_u8 整体标量化成逐字节
// 加载，性能下降一个数量级。构建系统对本文件显式传入
// -mno-strict-align 规避（用户态访问普通内存允许非对齐，安全）。
// 详见 CMakeLists.txt 中的说明。

#include <rse/config.hpp>

#if RSE_ARCH_AARCH64

#include <arm_neon.h>

#include "gf8_simd_kernel.hpp"

namespace rse::gf8::detail {

namespace {

struct NeonOps {
    using Vec = uint8x16_t;
    static constexpr std::size_t WIDTH = 16;

    static Vec load(const std::uint8_t* p) { return vld1q_u8(p); }
    static void store(std::uint8_t* p, Vec v) { vst1q_u8(p, v); }
    // 128 位向量恰好装下一张 16 字节查找表，无需广播。
    static Vec load_table(const std::uint8_t* p) { return vld1q_u8(p); }
    static Vec splat(std::uint8_t c) { return vdupq_n_u8(c); }
    static Vec and_(Vec a, Vec b) { return vandq_u8(a, b); }
    static Vec xor_(Vec a, Vec b) { return veorq_u8(a, b); }
    // NEON 直接支持按字节右移（不像 x86 需借助 64 位通道移位）。
    static Vec shift_right_4(Vec v) { return vshrq_n_u8(v, 4); }
    // TBL：越界下标返回 0，此处下标恒 < 16 不会发生。
    static Vec lookup(Vec table, Vec idx) { return vqtbl1q_u8(table, idx); }
};

} // namespace

SimdKernels neon_kernels() noexcept { return make_kernels<NeonOps>("neon"); }

} // namespace rse::gf8::detail

#endif // RSE_ARCH_AARCH64
