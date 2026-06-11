// test_galois_16.cpp - GF(2^16) 伽罗瓦域运算测试，
// 移植自 Rust 版 src/galois_16.rs 的 tests 模块
// （quickcheck 属性测试在这里以随机输入循环运行）。
//
// GF(2^16) 实现为 GF(2^8) 上的二次扩域，元素由两个字节组成
// （rse::gf16::Element），测试验证其域公理与幂运算性质。
#include "vendor/doctest.h"

#include <stdexcept>

#include <rse/galois_16.hpp>

#include "test_helpers.hpp"

using rse::gf16::Element;

namespace {

// 每个 quickcheck 风格测试的随机迭代次数。
constexpr int QC_ITERS = 200;

// 生成一个随机的 GF(2^16) 元素（高低两个字节均随机）。
Element random_element() {
    return Element(rse::test::random_u8(), rse::test::random_u8());
}

} // namespace

// 加法结合律：a + (b + c) == (a + b) + c。
// 对应 Rust 版的 qc_add_associativity。
TEST_CASE("qc: add associativity") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element(), b = random_element(), c = random_element();
        CHECK(a + (b + c) == (a + b) + c);
    }
}

// 乘法结合律：a * (b * c) == (a * b) * c。
// 对应 Rust 版的 qc_mul_associativity。
TEST_CASE("qc: mul associativity") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element(), b = random_element(), c = random_element();
        CHECK(a * (b * c) == (a * b) * c);
    }
}

// 加法单位元：a 减去其加法逆元（0 - a）应得到 0。
// 对应 Rust 版的 qc_additive_identity。
TEST_CASE("qc: additive identity") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element();
        const auto zero = Element::zero();
        CHECK(a - (zero - a) == zero);
    }
}

// 乘法单位元：非零元素 a 满足 (1 / a) * a == 1。
// 对应 Rust 版的 qc_multiplicative_identity。
TEST_CASE("qc: multiplicative identity") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element();
        if (a.is_zero()) continue;
        const auto one = Element(0, 1);
        CHECK((one / a) * a == one);
    }
}

// 加法交换律：a + b == b + a。对应 Rust 版的 qc_add_commutativity。
TEST_CASE("qc: add commutativity") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element(), b = random_element();
        CHECK(a + b == b + a);
    }
}

// 乘法交换律：a * b == b * a。对应 Rust 版的 qc_mul_commutativity。
TEST_CASE("qc: mul commutativity") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element(), b = random_element();
        CHECK(a * b == b * a);
    }
}

// 乘法对加法的分配律：a * (b + c) == a*b + a*c。
// 对应 Rust 版的 qc_add_distributivity。
TEST_CASE("qc: add distributivity") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element(), b = random_element(), c = random_element();
        CHECK(a * (b + c) == (a * b) + (a * c));
    }
}

// 乘法逆元：非零元素 a 与 a.inverse() 之积应为常量 1。
// 对应 Rust 版的 qc_inverse。
TEST_CASE("qc: inverse") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element();
        if (a.is_zero()) continue;
        const auto inv = a.inverse();
        CHECK(a * inv == Element::constant(1));
    }
}

// 幂运算性质 1：a.exp(n) 连续除以 a 共 (n-1) 次后应回到 a。
// 对应 Rust 版的 qc_exponent_1。
TEST_CASE("qc: exponent 1") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element();
        const auto n = static_cast<std::uint8_t>(rse::test::random_size(64));
        if (a.is_zero() || n == 0) continue;
        auto b = a.exp(n);
        for (int j = 1; j < n; ++j) {
            b = b / a;
        }
        CHECK(a == b);
    }
}

// 幂运算性质 2：从 1 开始反复乘以 a，第 j 步应等于 a.exp(j)。
// 对应 Rust 版的 qc_exponent_2。
TEST_CASE("qc: exponent 2") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element();
        const auto n = static_cast<std::uint8_t>(rse::test::random_size(64));
        if (a.is_zero()) continue;
        auto b = Element::constant(1);
        for (std::uint8_t j = 0; j < n; ++j) {
            CHECK(b == a.exp(j));
            b = b * a;
        }
    }
}

// 任意元素的 0 次幂均为常量 1。对应 Rust 版的 qc_exp_zero_is_one。
TEST_CASE("qc: exp zero is one") {
    for (int i = 0; i < QC_ITERS; ++i) {
        const auto a = random_element();
        CHECK(a.exp(0) == Element::constant(1));
    }
}

// 除数为零应抛出 std::domain_error。
// 对应 Rust 版的 div_b_is_0（Rust 中以 #[should_panic] 表达）。
TEST_CASE("div b is 0 throws") {
    CHECK_THROWS_AS((void)(Element(1, 0) / Element::zero()), std::domain_error);
}

// 零元素的 0 次幂按约定为 1。对应 Rust 版的 zero_to_zero_is_one。
TEST_CASE("zero to zero is one") { CHECK(Element::zero().exp(0) == Element::constant(1)); }
