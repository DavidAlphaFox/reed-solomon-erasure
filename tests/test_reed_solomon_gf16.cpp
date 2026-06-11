// test_reed_solomon_gf16.cpp - GF(2^16) 编解码器
// （rse::galois_16::ReedSolomon）的测试，
// 移植自 Rust 版 src/tests/galois_16.rs。
//
// GF(2^16) 的 shard 元素是两字节对（[u8; 2]），shard 总数上限为 2^16。
// quickcheck 属性测试在这里以随机输入循环运行；为控制耗时，
// 迭代次数与 shard 尺寸均比 GF(2^8) 测试更小。
#include "vendor/doctest.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <rse/rse.hpp>

#include "test_helpers.hpp"

using ReedSolomon16 = rse::galois_16::ReedSolomon;
using Elem = std::array<std::uint8_t, 2>;
using Shards16 = std::vector<std::vector<Elem>>;

using namespace rse::test;

namespace {

// 每个 quickcheck 风格测试的随机迭代次数（GF(2^16) 较慢，取小值）。
constexpr int QC_ITERS = 10;

} // namespace

// 验证 shard 总数受域阶（2^16）限制：data 或 parity 单边达到 ORDER
// 时创建失败，总数恰好等于 ORDER 时成功。
// 对应 Rust 版 src/tests/galois_16.rs 中的 correct_field_order_restriction。
TEST_CASE("correct field order restriction") {
    constexpr std::size_t ORDER = 1 << 16;

    CHECK(!ReedSolomon16::create(ORDER, 1).has_value());
    CHECK(!ReedSolomon16::create(1, ORDER).has_value());

    // (ORDER - 1, 1) 这一组合太慢，因为它需要构造 65536x65535 的
    // vandermonde 矩阵——与 Rust 版测试的处理一致，这里跳过它。
    CHECK(ReedSolomon16::create(1, ORDER - 1).has_value());
}

// quickcheck 属性：encode -> 随机破坏并清除 present 标志 ->
// reconstruct（present 标志形态）-> 验证与原始数据一致。
// 对应 Rust 版 galois_16.rs 的 qc_encode_verify_reconstruct_verify。
TEST_CASE("gf16 qc: encode verify reconstruct verify (flagged)") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params(256);
        const auto corrupt_pos_s = random_distinct_positions(p.corrupt, p.data + p.parity);

        const auto r = ReedSolomon16::create(p.data, p.parity).value();

        auto expect = make_random_shards<Elem>(p.size, p.data + p.parity);
        REQUIRE(r.encode(expect));

        auto shards = expect;
        const auto present = std::make_unique<bool[]>(p.data + p.parity);
        std::fill_n(present.get(), p.data + p.parity, true);
        for (const auto pos : corrupt_pos_s) {
            fill_random(std::span<Elem>(shards[pos]));
            present[pos] = false;
        }

        REQUIRE(r.reconstruct(shards, std::span(present.get(), p.data + p.parity)));

        CHECK(r.verify(expect).value());
        CHECK(expect == shards);
        CHECK(r.verify(shards).value());
    }
}

// 同上，但使用 std::optional 分片形态（缺失分片置为 nullopt）。
// 对应 Rust 版 galois_16.rs 的 qc_encode_verify_reconstruct_verify_shards。
TEST_CASE("gf16 qc: encode verify reconstruct verify (option shards)") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params(256);
        const auto corrupt_pos_s = random_distinct_positions(p.corrupt, p.data + p.parity);

        const auto r = ReedSolomon16::create(p.data, p.parity).value();

        auto expect = make_random_shards<Elem>(p.size, p.data + p.parity);
        REQUIRE(r.encode(expect));

        auto shards = shards_to_option_shards(expect);
        for (const auto pos : corrupt_pos_s) {
            shards[pos] = std::nullopt;
        }

        REQUIRE(r.reconstruct(shards));

        const auto plain = option_shards_to_shards(shards);
        CHECK(r.verify(expect).value());
        CHECK(expect == plain);
        CHECK(r.verify(plain).value());
    }
}

// quickcheck 属性：verify 的结果应与"shard 内容是否真的被改变"一致
// （随机破坏偶然等于原值的情形以一致性检查容忍）。
// 对应 Rust 版 galois_16.rs 的 qc_verify。
TEST_CASE("gf16 qc: verify") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params(256);
        const auto corrupt_pos_s = random_distinct_positions(p.corrupt, p.data + p.parity);

        const auto r = ReedSolomon16::create(p.data, p.parity).value();

        auto expect = make_random_shards<Elem>(p.size, p.data + p.parity);
        REQUIRE(r.encode(expect));

        auto shards = expect;
        for (const auto pos : corrupt_pos_s) {
            fill_random(std::span<Elem>(shards[pos]));
        }

        CHECK(r.verify(expect).value());
        const bool same = expect == shards;
        CHECK(r.verify(shards).value() == same);
        if (p.corrupt == 0) {
            CHECK(same);
        }
    }
}

// quickcheck 属性：encode_sep 的结果应与 encode 完全一致。
// 对应 Rust 版 galois_16.rs 的 qc_encode_sep_same_as_encode。
TEST_CASE("gf16 qc: encode_sep same as encode") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params(256);
        const auto r = ReedSolomon16::create(p.data, p.parity).value();

        auto expect = make_random_shards<Elem>(p.size, p.data + p.parity);
        auto shards = expect;

        REQUIRE(r.encode(expect));

        const auto data = std::span(shards).first(p.data);
        const auto parity = std::span(shards).subspan(p.data);
        REQUIRE(r.encode_sep(data, parity));

        CHECK(expect == shards);
    }
}

// quickcheck 属性：逐个调用 encode_single 的结果应与 encode 一致。
// 对应 Rust 版 galois_16.rs 的 qc_encode_single_same_as_encode。
TEST_CASE("gf16 qc: encode_single same as encode") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params(256);
        const auto r = ReedSolomon16::create(p.data, p.parity).value();

        auto expect = make_random_shards<Elem>(p.size, p.data + p.parity);
        auto shards = expect;

        REQUIRE(r.encode(expect));

        for (std::size_t i = 0; i < p.data; ++i) {
            REQUIRE(r.encode_single(i, shards));
        }

        CHECK(expect == shards);
    }
}

// quickcheck 属性：逐个调用 encode_single_sep 的结果应与 encode 一致。
// 对应 Rust 版 galois_16.rs 的 qc_encode_single_sep_same_as_encode。
TEST_CASE("gf16 qc: encode_single_sep same as encode") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params(256);
        const auto r = ReedSolomon16::create(p.data, p.parity).value();

        auto expect = make_random_shards<Elem>(p.size, p.data + p.parity);
        auto shards = expect;

        REQUIRE(r.encode(expect));

        const auto data = std::span(shards).first(p.data);
        const auto parity = std::span(shards).subspan(p.data);
        for (std::size_t i = 0; i < p.data; ++i) {
            REQUIRE(r.encode_single_sep(i, data[i], parity));
        }

        CHECK(expect == shards);
    }
}
