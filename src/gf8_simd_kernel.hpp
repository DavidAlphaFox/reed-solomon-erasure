// gf8_simd_kernel.hpp - GF(2^8) SIMD 乘法内核的公共算法模板（内部）。
//
// 算法只在这里写一次；各 ISA 翻译单元（gf8_simd_*.cpp）仅提供一组
// 满足下述要求的"向量操作原语"（Ops），由模板组合成完整内核——
// 这取代了重构前四份近乎相同的内核实现。
//
// 算法（"半字节拆分 + 字节 shuffle"，出自 Plank et al. 的
// "Screaming Fast Galois Field Arithmetic Using Intel SIMD Instructions"，
// 参考实现版权归 Nicolas Trangez (c) 2015,2016 与 Klaus Post (c) 2015，
// MIT 许可，见仓库根目录 LICENSE）：
//   1. 把每个输入字节 b 拆成高低两个半字节：lo = b & 0x0F，hi = b >> 4；
//   2. 以 lo / hi 为下标并行查 low / high 两张 16 项表（PSHUFB/TBL），
//      得到 c*lo 和 c*(hi<<4)；
//   3. 两者异或即 c*b（GF(2^8) 乘法对异或满足分配律）。
//
// Ops 需要提供：
//   using Vec               向量类型
//   static constexpr std::size_t WIDTH    向量宽度（字节）
//   static Vec load(const u8*)            非对齐加载一个向量
//   static void store(u8*, Vec)           非对齐存储一个向量
//   static Vec load_table(const u8*)      加载 16 字节查找表
//                                         （宽于 128 位的 ISA 需广播到各通道）
//   static Vec splat(u8)                  全字节广播常量
//   static Vec and_(Vec, Vec)             按位与
//   static Vec xor_(Vec, Vec)             按位异或
//   static Vec shift_right_4(Vec)         每字节逻辑右移 4 位；允许以
//                                         64 位通道粒度实现（如 x86 PSRLQ），
//                                         跨字节窜入的位会被随后的掩码清掉
//   static Vec lookup(Vec table, Vec idx) 16 项字节查表（idx 各字节 < 16）
#pragma once

#include <cstddef>
#include <cstdint>

#include "galois_simd.hpp"

namespace rse::gf8::detail {

// 半字节掩码：保留每个字节的低 4 位。
inline constexpr std::uint8_t LOW_NIBBLE_MASK = 0x0F;

// 内核模板。XOR 为 false 时覆盖写 out，为 true 时与 out 原值异或（乘加）。
// 两个变体由同一份代码编译期展开，无运行时分支。
// 返回已处理的字节数（WIDTH 的整数倍）；尾部由调用方标量补齐。
template <class Ops, bool XOR>
std::size_t gal_mul(const std::uint8_t* low, const std::uint8_t* high, const std::uint8_t* in,
                    std::uint8_t* out, std::size_t len) {
    const auto mask = Ops::splat(LOW_NIBBLE_MASK);
    const auto low_table = Ops::load_table(low);
    const auto high_table = Ops::load_table(high);

    std::size_t done = 0;
    for (std::size_t x = 0; x < len / Ops::WIDTH; ++x) {
        const auto in_x = Ops::load(in + done);
        const auto lo = Ops::and_(in_x, mask);
        const auto hi = Ops::and_(Ops::shift_right_4(in_x), mask);
        auto result = Ops::xor_(Ops::lookup(low_table, lo), Ops::lookup(high_table, hi));
        if constexpr (XOR) {
            result = Ops::xor_(result, Ops::load(out + done));
        }
        Ops::store(out + done, result);
        done += Ops::WIDTH;
    }
    return done;
}

// 由一组 Ops 生成成对的（覆盖写 + 异或累加）内核。
template <class Ops>
[[nodiscard]] SimdKernels make_kernels(const char* name) noexcept {
    return {&gal_mul<Ops, false>, &gal_mul<Ops, true>, name};
}

} // namespace rse::gf8::detail
