// test_galois_8.cpp - GF(2^8) 伽罗瓦域运算测试，
// 移植自 Rust 版 src/galois_8.rs 的 tests 模块
// （其中的 quickcheck 属性测试在这里以带种子的随机输入循环运行）。
//
// 内容包括：对数表与 Backblaze 参考实现的比对、域公理（结合律、
// 交换律、分配律、单位元）的穷举验证、已知向量测试，
// 以及 C++ 版新增的 SIMD 内核与标量参考实现的一致性测试。
#include "vendor/doctest.h"

#include <cstdint>
#include <vector>

#include <rse/galois_8.hpp>

#include "test_helpers.hpp"

namespace gf8 = rse::gf8;
using rse::test::fill_random;

namespace {

// 每个 quickcheck 风格测试的随机迭代次数（对应 quickcheck 的默认用例数）。
constexpr int QC_ITERS = 100;

// Backblaze 参考实现的对数表，用于交叉验证本库生成的 LOG_TABLE。
// 首个值由 -1 改为 0（与 Rust 版测试一致）。
constexpr std::uint8_t BACKBLAZE_LOG_TABLE[256] = {
    0,   0,   1,   25,  2,   50,  26,  198, 3,   223, 51,  238, 27,  104, 199, 75,  4,   100,
    224, 14,  52,  141, 239, 129, 28,  193, 105, 248, 200, 8,   76,  113, 5,   138, 101, 47,
    225, 36,  15,  33,  53,  147, 142, 218, 240, 18,  130, 69,  29,  181, 194, 125, 106, 39,
    249, 185, 201, 154, 9,   120, 77,  228, 114, 166, 6,   191, 139, 98,  102, 221, 48,  253,
    226, 152, 37,  179, 16,  145, 34,  136, 54,  208, 148, 206, 143, 150, 219, 189, 241, 210,
    19,  92,  131, 56,  70,  64,  30,  66,  182, 163, 195, 72,  126, 110, 107, 58,  40,  84,
    250, 133, 186, 61,  202, 94,  155, 159, 10,  21,  121, 43,  78,  212, 229, 172, 115, 243,
    167, 87,  7,   112, 192, 247, 140, 128, 99,  13,  103, 74,  222, 237, 49,  197, 254, 24,
    227, 165, 153, 119, 38,  184, 180, 124, 17,  68,  146, 217, 35,  32,  137, 46,  55,  63,
    209, 91,  149, 188, 207, 205, 144, 135, 151, 178, 220, 252, 190, 97,  242, 86,  211, 171,
    20,  42,  93,  158, 132, 60,  57,  83,  71,  109, 65,  162, 31,  45,  67,  216, 183, 123,
    164, 118, 196, 23,  73,  236, 127, 12,  111, 246, 108, 161, 59,  82,  41,  157, 85,  170,
    251, 96,  134, 177, 187, 204, 62,  90,  203, 89,  95,  176, 156, 169, 160, 81,  11,  245,
    22,  235, 122, 117, 44,  215, 79,  174, 213, 233, 230, 231, 173, 232, 116, 214, 244, 234,
    168, 80,  88,  175,
};

} // namespace

// 验证本库的对数表与 Backblaze 实现逐项一致。
// 对应 Rust 版的 log_table_same_as_backblaze。
TEST_CASE("log table same as backblaze") {
    for (int i = 0; i < 256; ++i) {
        CHECK(gf8::LOG_TABLE[i] == BACKBLAZE_LOG_TABLE[i]);
    }
}

// 穷举全部 256^3 组合，验证 add 与 mul 满足结合律。
// 对应 Rust 版的 test_associativity。
TEST_CASE("associativity (exhaustive)") {
    for (int a = 0; a < 256; ++a) {
        for (int b = 0; b < 256; ++b) {
            for (int c = 0; c < 256; ++c) {
                const auto ua = static_cast<std::uint8_t>(a);
                const auto ub = static_cast<std::uint8_t>(b);
                const auto uc = static_cast<std::uint8_t>(c);
                if (gf8::add(ua, gf8::add(ub, uc)) != gf8::add(gf8::add(ua, ub), uc)) {
                    FAIL("add not associative");
                }
                if (gf8::mul(ua, gf8::mul(ub, uc)) != gf8::mul(gf8::mul(ua, ub), uc)) {
                    FAIL("mul not associative");
                }
            }
        }
    }
}

// 验证加法逆元（自身即逆元）与乘法逆元（1/a）的单位元性质。
// 对应 Rust 版的 test_identity。
TEST_CASE("identity") {
    for (int a = 0; a < 256; ++a) {
        const auto ua = static_cast<std::uint8_t>(a);
        const auto b = gf8::sub(0, ua);
        CHECK(gf8::sub(ua, b) == 0);
        if (ua != 0) {
            const auto inv = gf8::div(1, ua);
            CHECK(gf8::mul(ua, inv) == 1);
        }
    }
}

// 穷举验证 add 与 mul 满足交换律。
// 对应 Rust 版的 test_commutativity。
TEST_CASE("commutativity") {
    for (int a = 0; a < 256; ++a) {
        for (int b = 0; b < 256; ++b) {
            const auto ua = static_cast<std::uint8_t>(a);
            const auto ub = static_cast<std::uint8_t>(b);
            if (gf8::add(ua, ub) != gf8::add(ub, ua)) FAIL("add not commutative");
            if (gf8::mul(ua, ub) != gf8::mul(ub, ua)) FAIL("mul not commutative");
        }
    }
}

// 穷举验证乘法对加法的分配律。
// 对应 Rust 版的 test_distributivity。
TEST_CASE("distributivity (exhaustive)") {
    for (int a = 0; a < 256; ++a) {
        for (int b = 0; b < 256; ++b) {
            for (int c = 0; c < 256; ++c) {
                const auto ua = static_cast<std::uint8_t>(a);
                const auto ub = static_cast<std::uint8_t>(b);
                const auto uc = static_cast<std::uint8_t>(c);
                if (gf8::mul(ua, gf8::add(ub, uc)) !=
                    gf8::add(gf8::mul(ua, ub), gf8::mul(ua, uc))) {
                    FAIL("mul not distributive over add");
                }
            }
        }
    }
}

// 验证 exp(a, j) 等于 a 连乘 j 次，覆盖全部底数与 0..255 的指数。
// 对应 Rust 版的 test_exp。
TEST_CASE("exp") {
    for (int a = 0; a < 256; ++a) {
        const auto ua = static_cast<std::uint8_t>(a);
        std::uint8_t power = 1;
        for (std::size_t j = 0; j < 256; ++j) {
            CHECK(gf8::exp(ua, j) == power);
            power = gf8::mul(power, ua);
        }
    }
}

// 已知向量测试：固定输入下验证 mul、mul_slice、mul_slice_xor、exp
// 的输出与预先计算好的期望值一致（每种调用执行两次以确认无副作用）。
// 对应 Rust 版的 test_galois。
TEST_CASE("galois known vectors") {
    CHECK(gf8::mul(3, 4) == 12);
    CHECK(gf8::mul(7, 7) == 21);
    CHECK(gf8::mul(23, 45) == 41);

    const std::vector<std::uint8_t> input = {
        0,   1,   2,   3,   4,   5,   6,   10,  50,  100, 150, 174, 201, 255, 99,  32,  67,
        85,  200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 188, 187, 186, 185,
    };
    std::vector<std::uint8_t> output1(input.size(), 0);
    std::vector<std::uint8_t> output2(input.size(), 0);

    gf8::mul_slice(25, input, output1);
    const std::vector<std::uint8_t> expect = {
        0x0,  0x19, 0x32, 0x2b, 0x64, 0x7d, 0x56, 0xfa, 0xb8, 0x6d, 0xc7, 0x85,
        0xc3, 0x1f, 0x22, 0x7,  0x25, 0xfe, 0xda, 0x5d, 0x44, 0x6f, 0x76, 0x39,
        0x20, 0xb,  0x12, 0x11, 0x8,  0x23, 0x3a, 0x75, 0x6c, 0x47,
    };
    CHECK(output1 == expect);
    gf8::mul_slice(25, input, output2);
    CHECK(output2 == expect);

    const std::vector<std::uint8_t> expect_xor = {
        0x0,  0x2d, 0x5a, 0x77, 0xb4, 0x99, 0xee, 0x2f, 0x79, 0xf2, 0x7,  0x51,
        0xd4, 0x19, 0x31, 0xc9, 0xf8, 0xfc, 0xf9, 0x4f, 0x62, 0x15, 0x38, 0xfb,
        0xd6, 0xa1, 0x8c, 0x96, 0xbb, 0xcc, 0xe1, 0x22, 0xf,  0x78,
    };
    gf8::mul_slice_xor(52, input, output1);
    CHECK(output1 == expect_xor);
    gf8::mul_slice_xor(52, input, output2);
    CHECK(output2 == expect_xor);

    const std::vector<std::uint8_t> expect2 = {
        0x0,  0xb1, 0x7f, 0xce, 0xfe, 0x4f, 0x81, 0x9e, 0x3,  0x6,  0xe8, 0x75,
        0xbd, 0x40, 0x36, 0xa3, 0x95, 0xcb, 0xc,  0xdd, 0x6c, 0xa2, 0x13, 0x23,
        0x92, 0x5c, 0xed, 0x1b, 0xaa, 0x64, 0xd5, 0xe5, 0x54, 0x9a,
    };
    gf8::mul_slice(177, input, output1);
    CHECK(output1 == expect2);
    gf8::mul_slice(177, input, output2);
    CHECK(output2 == expect2);

    const std::vector<std::uint8_t> expect_xor2 = {
        0x0,  0xc4, 0x95, 0x51, 0x37, 0xf3, 0xa2, 0xfb, 0xec, 0xc5, 0xd0, 0xc7,
        0x53, 0x88, 0xa3, 0xa5, 0x6,  0x78, 0x97, 0x9f, 0x5b, 0xa,  0xce, 0xa8,
        0x6c, 0x3d, 0xf9, 0xdf, 0x1b, 0x4a, 0x8e, 0xe8, 0x2c, 0x7d,
    };
    gf8::mul_slice_xor(117, input, output1);
    CHECK(output1 == expect_xor2);
    gf8::mul_slice_xor(117, input, output2);
    CHECK(output2 == expect_xor2);

    CHECK(gf8::exp(2, 2) == 4);
    CHECK(gf8::exp(5, 20) == 235);
    CHECK(gf8::exp(13, 7) == 43);
}

// 验证 slice_xor（切片逐字节异或，即 GF(2^8) 上的切片加法）：
// 多种长度（含非对齐的 34）下结果应等于逐字节 XOR 的标量计算。
// 对应 Rust 版的 test_slice_add。
TEST_CASE("slice add") {
    for (const std::size_t len : {16, 32, 34}) {
        std::vector<std::uint8_t> input(len);
        fill_random(input);
        std::vector<std::uint8_t> output(len);
        fill_random(output);

        std::vector<std::uint8_t> expect(len);
        for (std::size_t i = 0; i < len; ++i) {
            expect[i] = input[i] ^ output[i];
        }
        gf8::slice_xor(input, output);
        CHECK(output == expect);

        fill_random(output);
        for (std::size_t i = 0; i < len; ++i) {
            expect[i] = input[i] ^ output[i];
        }
        gf8::slice_xor(input, output);
        CHECK(output == expect);
    }
}

// 0 除以任意非零数应为 0。对应 Rust 版的 test_div_a_is_0。
TEST_CASE("div a is 0") { CHECK(gf8::div(0, 100) == 0); }

// 除数为 0 应抛出 std::domain_error。
// 对应 Rust 版的 test_div_b_is_0（Rust 中以 #[should_panic] 表达）。
TEST_CASE("div b is 0 throws") { CHECK_THROWS_AS((void)gf8::div(1, 0), std::domain_error); }

// 在 Rust crate 中这个测试是把 SIMD FFI 实现与重复调用的结果做比较；
// 这里它兼作运行时分发 SIMD 路径的确定性检查，
// 并采用奇数长度以覆盖标量尾部处理。
TEST_CASE("mul_slice deterministic over simd dispatch") {
    constexpr std::size_t len = 10'003;
    for (int round = 0; round < 100; ++round) {
        const std::uint8_t c = rse::test::random_u8();
        std::vector<std::uint8_t> input(len);
        fill_random(input);
        {
            std::vector<std::uint8_t> output(len);
            fill_random(output);
            auto output_copy = output;

            gf8::mul_slice(c, input, output);
            gf8::mul_slice(c, input, output_copy);
            CHECK(output == output_copy);
        }
        {
            std::vector<std::uint8_t> output(len);
            fill_random(output);
            auto output_copy = output;

            gf8::mul_slice_xor(c, input, output);
            gf8::mul_slice_xor(c, input, output_copy);
            CHECK(output == output_copy);
        }
    }
}

// C++ 版新增测试（Rust 版无对应）：SIMD 内核必须与标量参考实现
// 在全部 256 个乘数下逐字节一致；长度集合覆盖各向量宽度的边界
// （16/32/64 字节前后）以及标量尾部处理。
TEST_CASE("simd kernels match scalar reference") {
    INFO("active SIMD backend: ", gf8::simd_backend());
    for (const std::size_t len : {1u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 1000u}) {
        std::vector<std::uint8_t> input(len);
        fill_random(input);
        for (int c = 0; c < 256; ++c) {
            const auto uc = static_cast<std::uint8_t>(c);

            std::vector<std::uint8_t> simd_out(len);
            fill_random(simd_out);

            gf8::mul_slice(uc, input, simd_out);
            for (std::size_t i = 0; i < len; ++i) {
                if (simd_out[i] != gf8::mul(uc, input[i])) {
                    FAIL("mul_slice mismatch at len=", len, " c=", c, " i=", i);
                }
            }
        }
    }
}

// C++ 版新增测试（Rust 版无对应）：mul_slice_xor 的 SIMD 路径
// 必须与"原值 XOR 标量乘积"的参考结果一致，覆盖全部乘数和
// 跨越向量宽度边界的多种长度。
TEST_CASE("mul_slice_xor matches scalar reference") {
    for (const std::size_t len : {1u, 16u, 33u, 64u, 65u, 1000u}) {
        std::vector<std::uint8_t> input(len);
        fill_random(input);
        for (int c = 0; c < 256; ++c) {
            const auto uc = static_cast<std::uint8_t>(c);
            std::vector<std::uint8_t> out(len);
            fill_random(out);
            const auto orig = out;

            gf8::mul_slice_xor(uc, input, out);
            for (std::size_t i = 0; i < len; ++i) {
                if (out[i] != static_cast<std::uint8_t>(orig[i] ^ gf8::mul(uc, input[i]))) {
                    FAIL("mul_slice_xor mismatch at len=", len, " c=", c, " i=", i);
                }
            }
        }
    }
}

// quickcheck 属性测试（随机输入循环）：随机抽取 a、b、c，
// 验证结合律、交换律、分配律以及加法/乘法单位元性质。
// 汇总对应 Rust 版的 qc_add_associativity、qc_mul_associativity、
// qc_add_commutativity、qc_mul_commutativity、qc_add_distributivity、
// qc_additive_identity、qc_multiplicative_identity。
TEST_CASE("qc: add/mul associativity, commutativity, distributivity, identities") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const std::uint8_t a = rse::test::random_u8();
        const std::uint8_t b = rse::test::random_u8();
        const std::uint8_t c = rse::test::random_u8();

        CHECK(gf8::add(a, gf8::add(b, c)) == gf8::add(gf8::add(a, b), c));
        CHECK(gf8::mul(a, gf8::mul(b, c)) == gf8::mul(gf8::mul(a, b), c));
        CHECK(gf8::add(a, b) == gf8::add(b, a));
        CHECK(gf8::mul(a, b) == gf8::mul(b, a));
        CHECK(gf8::mul(a, gf8::add(b, c)) == gf8::add(gf8::mul(a, b), gf8::mul(a, c)));
        CHECK(gf8::sub(a, gf8::sub(0, a)) == 0);
        if (a != 0) {
            CHECK(gf8::mul(a, gf8::div(1, a)) == 1);
        }
    }
}
