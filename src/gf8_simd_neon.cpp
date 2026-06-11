// gf8_simd_neon.cpp - ARM NEON（128 位）GF(2^8) 乘法内核。
//
// 移植自 simd_c/reedsolomon.c 的 NEON 路径（版权归 Nicolas Trangez 与
// Klaus Post 所有，MIT 许可，见仓库根目录 LICENSE）。
//
// 算法与 x86 各版相同（半字节拆分 + 字节查表，详见
// gf8_simd_ssse3.cpp 的注释）；查表指令用 AArch64 的 TBL
// （vqtbl1q_u8，与 x86 PSHUFB 等价的 16 字节并行查表）。
// NEON 是 AArch64 的基线特性，无需特殊编译选项与运行时检测。
//
// 移植注意事项：某些工具链（如 OpenBSD clang）在 AArch64 上默认
// +strict-align，会把非对齐向量加载 vld1q_u8 整体标量化成逐字节
// 加载，性能下降一个数量级。构建系统对本文件显式传入
// -mno-strict-align 规避（用户态访问普通内存允许非对齐，安全）。
// 详见 CMakeLists.txt 中的说明。

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include "galois_simd.hpp"

namespace rse::gf8::detail {

namespace {

// XOR 含义同 x86 各版：false 覆盖写，true 异或累加。
template <bool XOR>
std::size_t gal_mul_neon(const std::uint8_t* low, const std::uint8_t* high,
                         const std::uint8_t* in, std::uint8_t* out, std::size_t len) {
    const uint8x16_t low_mask = vdupq_n_u8(0x0f);
    // 两张 16 字节查找表各装入一个向量寄存器。
    const uint8x16_t low_vector = vld1q_u8(low);
    const uint8x16_t high_vector = vld1q_u8(high);

    std::size_t done = 0;
    for (std::size_t x = 0; x < len / 16; ++x) {
        const uint8x16_t in_x = vld1q_u8(in + done);
        // vshrq_n_u8 按字节右移（不像 x86 的 PSRLQ 需要再掩位，
        // 这里 & 0x0F 只是保持与参考实现一致的防御性写法）。
        const uint8x16_t low_input = vandq_u8(in_x, low_mask);
        const uint8x16_t high_input = vandq_u8(vshrq_n_u8(in_x, 4), low_mask);
        // TBL 并行查表（越界下标返回 0，此处下标恒 < 16 不会发生）。
        const uint8x16_t mul_low = vqtbl1q_u8(low_vector, low_input);
        const uint8x16_t mul_high = vqtbl1q_u8(high_vector, high_input);
        uint8x16_t result = veorq_u8(mul_low, mul_high);
        if constexpr (XOR) {
            result = veorq_u8(result, vld1q_u8(out + done));
        }
        vst1q_u8(out + done, result);
        done += 16;
    }
    return done; // 尾部（< 16 字节）由调用方标量处理
}

} // namespace

SimdKernels neon_kernels() noexcept {
    return {&gal_mul_neon<false>, &gal_mul_neon<true>, "neon"};
}

} // namespace rse::gf8::detail

#endif // aarch64
