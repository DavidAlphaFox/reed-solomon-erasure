// test_reed_solomon.cpp - GF(2^8) 编解码器（rse::galois_8::ReedSolomon）
// 与 ShardByShard 逐分片编码器的测试，移植自 Rust 版 src/tests/mod.rs。
//
// Rust 版中许多测试存在两份：一份针对 `Vec<Vec<u8>>`（"shards"），
// 一份针对 `&mut [u8]` 引用（"slices"）；C++ 版两者统一为同一套
// 基于 range 的 API，因此每对测试只移植一次。quickcheck 属性测试
// 以随机输入循环运行。
//
// 错误均通过 std::expected 返回：编解码器返回 rse::Error，
// ShardByShard 返回 SBSError。reconstruct 有两种形态：
// std::optional 分片（缺失分片为 nullopt），以及
// 缓冲区 + present 布尔标志（标记每个分片是否有效）。
#include "vendor/doctest.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <rse/rse.hpp>

#include "test_helpers.hpp"

using rse::Error;
using rse::SBSError;
using ReedSolomon = rse::galois_8::ReedSolomon;
using ShardByShard = rse::galois_8::ShardByShard;
using Shards = std::vector<std::vector<std::uint8_t>>;
using OptionShards = std::vector<std::optional<std::vector<std::uint8_t>>>;

using namespace rse::test;

namespace {

// 每个 quickcheck 风格测试的随机迭代次数。
constexpr int QC_ITERS = 20;

// 逐分片断言两个 shard 集合完全相等（先比较数量，再逐个比较内容）。
// 对应 Rust 版 src/tests/mod.rs 中的 assert_eq_shards。
void check_eq_shards(const Shards& s1, const Shards& s2) {
    REQUIRE(s1.size() == s2.size());
    for (std::size_t i = 0; i < s1.size(); ++i) {
        CHECK(s1[i] == s2[i]);
    }
}

} // namespace

// data shard 数为 0 时创建编解码器应失败。
// 对应 Rust 版的 test_no_data_shards。
TEST_CASE("no data shards") {
    CHECK(ReedSolomon::create(0, 1).error() == Error::TooFewDataShards);
}

// parity shard 数为 0 时创建编解码器应失败。
// 对应 Rust 版的 test_no_parity_shards。
TEST_CASE("no parity shards") {
    CHECK(ReedSolomon::create(1, 0).error() == Error::TooFewParityShards);
}

// shard 总数超过 GF(2^8) 的上限 256 时创建应失败。
// 对应 Rust 版的 test_too_many_shards。
TEST_CASE("too many shards") {
    CHECK(ReedSolomon::create(129, 128).error() == Error::TooManyShards);
}

// 随机参数下验证 data/parity/total shard 数量查询接口。
// 对应 Rust 版的 test_shard_count。
TEST_CASE("shard count") {
    for (int i = 0; i < 10; ++i) {
        const std::size_t data_shard_count = 1 + random_size(127);
        const std::size_t parity_shard_count = 1 + random_size(127);

        const auto r = ReedSolomon::create(data_shard_count, parity_shard_count).value();

        CHECK(r.data_shard_count() == data_shard_count);
        CHECK(r.parity_shard_count() == parity_shard_count);
        CHECK(r.total_shard_count() == data_shard_count + parity_shard_count);
    }
}

// 编解码器拷贝后应与原对象相等。
// 对应 Rust 版的 test_reed_solomon_clone（Clone 在 C++ 中即拷贝构造）。
TEST_CASE("reed solomon copy") {
    const auto r1 = ReedSolomon::create(10, 3).value();
    const ReedSolomon r2 = r1; // NOLINT(performance-unnecessary-copy-initialization)
    CHECK(r1 == r2);
}

// 基本编码流程：encode 后 verify 通过；shard 数不足或尺寸不一致时
// 返回相应错误。对应 Rust 版的 test_encoding。
TEST_CASE("encoding") {
    constexpr std::size_t per_shard = 50'000;

    const auto r = ReedSolomon::create(10, 3).value();

    auto shards = make_random_shards(per_shard, 13);

    REQUIRE(r.encode(shards));
    CHECK(r.verify(shards).value());

    CHECK(r.encode(std::span(shards).first(1)).error() == Error::TooFewShards);

    auto bad_shards = make_random_shards(per_shard, 13);
    bad_shards[0] = std::vector<std::uint8_t>{0};
    CHECK(r.encode(bad_shards).error() == Error::IncorrectShardSize);
}

// 使用 std::optional 分片形态的重建流程：在不同缺失组合下
// reconstruct / reconstruct_data，并验证重建结果与原始数据一致。
// 对应 Rust 版的 test_reconstruct_shards。
TEST_CASE("reconstruct shards") {
    constexpr std::size_t per_shard = 100'000;

    const auto r = ReedSolomon::create(8, 5).value();

    auto master = make_random_shards(per_shard, 13);
    REQUIRE(r.encode(master));

    auto shards = shards_to_option_shards(master);

    // 所有 shard 均在场时尝试解码（应为无操作的成功）。
    REQUIRE(r.reconstruct(shards));
    {
        const auto plain = option_shards_to_shards(shards);
        CHECK(r.verify(plain).value());
        check_eq_shards(plain, master);
    }

    // 仅剩 10 个 shard（缺 2 个）时尝试解码。
    shards[0] = std::nullopt;
    shards[2] = std::nullopt;
    REQUIRE(r.reconstruct(shards));
    {
        const auto plain = option_shards_to_shards(shards);
        CHECK(r.verify(plain).value());
        check_eq_shards(plain, master);
    }

    // 以相同的缺失组合再次解码，命中缓存的 decode matrix（LruCache）。
    shards[0] = std::nullopt;
    shards[2] = std::nullopt;
    REQUIRE(r.reconstruct(shards));
    {
        const auto plain = option_shards_to_shards(shards);
        CHECK(r.verify(plain).value());
        check_eq_shards(plain, master);
    }

    // 仅剩 6 个 data shard 和 4 个 parity shard 时尝试解码。
    shards[0] = std::nullopt;
    shards[2] = std::nullopt;
    shards[12] = std::nullopt;
    REQUIRE(r.reconstruct(shards));
    {
        const auto plain = option_shards_to_shards(shards);
        CHECK(r.verify(plain).value());
        check_eq_shards(plain, master);
    }

    // 只重建 data shard（reconstruct_data 不补回缺失的 parity shard）。
    shards[0] = std::nullopt;
    shards[1] = std::nullopt;
    shards[12] = std::nullopt;
    REQUIRE(r.reconstruct_data(shards));
    {
        CHECK(shards[0].value() == master[0]);
        CHECK(shards[1].value() == master[1]);
        CHECK(!shards[12].has_value());
    }

    // 仅剩 7 个 shard（6 data + 1 parity）：在场数量少于 data shard 数 8，
    // 无法重建，应返回 TooFewShardsPresent。
    shards[0] = std::nullopt;
    shards[1] = std::nullopt;
    shards[9] = std::nullopt;
    shards[10] = std::nullopt;
    shards[11] = std::nullopt;
    shards[12] = std::nullopt;
    CHECK(r.reconstruct(shards).error() == Error::TooFewShardsPresent);
}

// 使用"缓冲区 + present 标志"形态的重建，输入输出均为已知向量：
// 依次验证 reconstruct 全量重建与 reconstruct_data 仅重建 data shard
// 的行为。对应 Rust 版的 test_reconstruct。
TEST_CASE("reconstruct (flagged, known vectors)") {
    const auto r = ReedSolomon::create(2, 2).value();

    Shards shards = {{0, 1, 2}, {3, 4, 5}, {200, 201, 203}, {100, 101, 102}};

    REQUIRE(r.encode(shards));
    CHECK(r.verify(shards).value());

    {
        shards[0] = {101, 102, 103};
        const bool present[] = {false, true, true, true};
        REQUIRE(r.reconstruct(shards, present));
        CHECK(r.verify(shards).value());
    }
    const Shards expect1 = {{0, 1, 2}, {3, 4, 5}, {6, 11, 12}, {5, 14, 11}};
    check_eq_shards(shards, expect1);

    {
        shards[0] = {201, 202, 203};
        shards[2] = {101, 102, 103};
        const bool present[] = {false, true, false, true};
        REQUIRE(r.reconstruct_data(shards, present));
        CHECK(!r.verify(shards).value());
    }
    const Shards expect2 = {{0, 1, 2}, {3, 4, 5}, {101, 102, 103}, {5, 14, 11}};
    check_eq_shards(shards, expect2);

    {
        shards[2] = {101, 102, 103};
        shards[3] = {201, 202, 203};
        const bool present[] = {true, true, false, false};
        REQUIRE(r.reconstruct_data(shards, present));
        CHECK(!r.verify(shards).value());
    }
    const Shards expect3 = {{0, 1, 2}, {3, 4, 5}, {101, 102, 103}, {201, 202, 203}};
    check_eq_shards(shards, expect3);
}

// quickcheck 属性（随机输入循环）：随机参数下 encode -> 随机破坏若干
// shard 并清除 present 标志 -> reconstruct（present 标志形态）->
// 结果应与原始数据一致且通过 verify。
// 对应 Rust 版的 qc_encode_verify_reconstruct_verify。
TEST_CASE("qc: encode verify reconstruct verify (flagged)") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params();
        const auto corrupt_pos_s = random_distinct_positions(p.corrupt, p.data + p.parity);

        const auto r = ReedSolomon::create(p.data, p.parity).value();

        auto expect = make_random_shards(p.size, p.data + p.parity);
        REQUIRE(r.encode(expect));

        auto shards = expect;
        const auto present = std::make_unique<bool[]>(p.data + p.parity);
        std::fill_n(present.get(), p.data + p.parity, true);
        for (const auto pos : corrupt_pos_s) {
            fill_random(std::span<std::uint8_t>(shards[pos]));
            present[pos] = false;
        }

        REQUIRE(r.reconstruct(shards, std::span(present.get(), p.data + p.parity)));

        CHECK(r.verify(expect).value());
        CHECK(expect == shards);
        CHECK(r.verify(shards).value());
    }
}

// 同上，但使用 std::optional 分片形态（缺失分片置为 nullopt）。
// 对应 Rust 版的 qc_encode_verify_reconstruct_verify_shards。
TEST_CASE("qc: encode verify reconstruct verify (option shards)") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params();
        const auto corrupt_pos_s = random_distinct_positions(p.corrupt, p.data + p.parity);

        const auto r = ReedSolomon::create(p.data, p.parity).value();

        auto expect = make_random_shards(p.size, p.data + p.parity);
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

// quickcheck 属性：未被破坏的数据 verify 应通过；被随机破坏后，
// verify 的结果应与"内容是否真的发生变化"一致。
// 对应 Rust 版的 qc_verify。
TEST_CASE("qc: verify") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params();
        const auto corrupt_pos_s = random_distinct_positions(p.corrupt, p.data + p.parity);

        const auto r = ReedSolomon::create(p.data, p.parity).value();

        auto expect = make_random_shards(p.size, p.data + p.parity);
        REQUIRE(r.encode(expect));

        auto shards = expect;
        for (const auto pos : corrupt_pos_s) {
            fill_random(std::span<std::uint8_t>(shards[pos]));
        }

        CHECK(r.verify(expect).value());
        if (p.corrupt > 0) {
            // 随机破坏可能恰好与原始数据相同；非平凡尺寸下概率可忽略，
            // 这里通过检查"相等性"与"verify 结果"的一致性来容忍该情形。
            const bool same = expect == shards;
            CHECK(r.verify(shards).value() == same);
        } else {
            CHECK(expect == shards);
            CHECK(r.verify(shards).value());
        }
    }
}

// quickcheck 属性：encode_sep（data 与 parity 分离传入）的结果
// 应与 encode 完全一致。对应 Rust 版的 qc_encode_sep_same_as_encode。
TEST_CASE("qc: encode_sep same as encode") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params();
        const auto r = ReedSolomon::create(p.data, p.parity).value();

        auto expect = make_random_shards(p.size, p.data + p.parity);
        auto shards = expect;

        REQUIRE(r.encode(expect));

        const auto data = std::span(shards).first(p.data);
        const auto parity = std::span(shards).subspan(p.data);
        REQUIRE(r.encode_sep(data, parity));

        CHECK(expect == shards);
    }
}

// quickcheck 属性：对每个 data shard 依次调用 encode_single，
// 最终结果应与一次性 encode 相同。
// 对应 Rust 版的 qc_encode_single_same_as_encode。
TEST_CASE("qc: encode_single same as encode") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params();
        const auto r = ReedSolomon::create(p.data, p.parity).value();

        auto expect = make_random_shards(p.size, p.data + p.parity);
        auto shards = expect;

        REQUIRE(r.encode(expect));

        for (std::size_t i = 0; i < p.data; ++i) {
            REQUIRE(r.encode_single(i, shards));
        }

        CHECK(expect == shards);
    }
}

// quickcheck 属性：encode_single_sep（单个 data shard + 分离的 parity）
// 逐个调用后结果应与 encode 相同。
// 对应 Rust 版的 qc_encode_single_sep_same_as_encode。
TEST_CASE("qc: encode_single_sep same as encode") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params();
        const auto r = ReedSolomon::create(p.data, p.parity).value();

        auto expect = make_random_shards(p.size, p.data + p.parity);
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

// present 标志形态下的错误处理：在场 shard 数不足时返回
// TooFewShardsPresent，恰好足够时成功。
// 对应 Rust 版的 test_reconstruct_error_handling。
TEST_CASE("reconstruct error handling") {
    const auto r = ReedSolomon::create(2, 2).value();

    Shards shards = {{0, 1, 2}, {3, 4, 5}, {200, 201, 203}, {100, 101, 102}};
    REQUIRE(r.encode(shards));

    shards[0] = {101, 102, 103};
    {
        const bool present[] = {true, false, false, false};
        CHECK(r.reconstruct(shards, present).error() == Error::TooFewShardsPresent);
    }
    {
        const bool present[] = {true, false, false, true};
        REQUIRE(r.reconstruct(shards, present));
    }
}

// 已知向量编码：5+5 配置下验证每个 parity 字节的精确值，
// 且篡改任一字节后 verify 应失败。对应 Rust 版的 test_one_encode。
TEST_CASE("one encode") {
    const auto r = ReedSolomon::create(5, 5).value();

    Shards shards = {{0, 1}, {4, 5}, {2, 3}, {6, 7}, {8, 9},
                     {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};

    REQUIRE(r.encode(shards));

    CHECK(shards[5][0] == 12);
    CHECK(shards[5][1] == 13);
    CHECK(shards[6][0] == 10);
    CHECK(shards[6][1] == 11);
    CHECK(shards[7][0] == 14);
    CHECK(shards[7][1] == 15);
    CHECK(shards[8][0] == 90);
    CHECK(shards[8][1] == 91);
    CHECK(shards[9][0] == 94);
    CHECK(shards[9][1] == 95);

    CHECK(r.verify(shards).value());

    shards[8][0] += 1;
    CHECK(!r.verify(shards).value());
}

// shard 数量不足时 verify 应返回 TooFewShards。
// 对应 Rust 版的 test_verify_too_few_shards。
TEST_CASE("verify too few shards") {
    const auto r = ReedSolomon::create(3, 2).value();

    const auto shards = make_random_shards(10, 4);
    CHECK(r.verify(shards).error() == Error::TooFewShards);
}

// verify_with_buffer 对缓冲区形参的各种尺寸校验：
// 缓冲区 shard 数量过少/过多、首个为空、长度不一致等错误路径。
// 对应 Rust 版的 test_verify_with_buffer_incorrect_buffer_sizes。
TEST_CASE("verify with buffer: incorrect buffer sizes") {
    const auto r = ReedSolomon::create(3, 2).value();

    {
        // 缓冲区中的切片数量过少。
        const auto shards = make_random_shards(100, 5);
        auto buffer = make_random_shards(100, 1);
        CHECK(r.verify_with_buffer(shards, buffer).error() == Error::TooFewBufferShards);
    }
    {
        // 缓冲区中的切片数量过多。
        const auto shards = make_random_shards(100, 5);
        auto buffer = make_random_shards(100, 3);
        CHECK(r.verify_with_buffer(shards, buffer).error() == Error::TooManyBufferShards);
    }
    {
        // 缓冲区中的切片数量正确（成功路径）。
        auto shards = make_random_shards(100, 5);
        REQUIRE(r.encode(shards));
        auto buffer = make_random_shards(100, 2);
        CHECK(r.verify_with_buffer(shards, buffer).value());
    }
    {
        // 缓冲区的首个 shard 为空。
        const auto shards = make_random_shards(100, 5);
        auto buffer = make_random_shards(100, 2);
        buffer[0].clear();
        CHECK(r.verify_with_buffer(shards, buffer).error() == Error::EmptyShard);
    }
    {
        // 缓冲区各 shard 长度不一致。
        const auto shards = make_random_shards(100, 5);
        auto buffer = make_random_shards(100, 2);
        buffer[1].resize(99);
        CHECK(r.verify_with_buffer(shards, buffer).error() == Error::IncorrectShardSize);
    }
}

// 无论 verify 结果是否通过，verify_with_buffer 都应把计算出的
// parity shard 写入缓冲区，且与 encode 产生的 parity 一致。
// 对应 Rust 版的 test_verify_with_buffer_gives_correct_parity_shards。
TEST_CASE("verify with buffer gives correct parity shards") {
    const auto r = ReedSolomon::create(10, 3).value();

    for (int round = 0; round < 100; ++round) {
        auto shards = make_random_shards(100, 13);
        const auto shards_copy = shards;

        REQUIRE(r.encode(shards));

        {
            auto buffer = make_random_shards(100, 3);
            CHECK(!r.verify_with_buffer(shards_copy, buffer).value());
            for (int i = 0; i < 3; ++i) {
                CHECK(shards[10 + i] == buffer[i]);
            }
        }
        {
            auto buffer = make_random_shards(100, 3);
            CHECK(r.verify_with_buffer(shards, buffer).value());
            for (int i = 0; i < 3; ++i) {
                CHECK(shards[10 + i] == buffer[i]);
            }
        }
    }
}

// shard 总数过少/过多时 encode、verify、reconstruct 均应返回
// 相应的数量错误。对应 Rust 版的 test_slices_or_shards_count_check。
TEST_CASE("slices or shards count check") {
    const auto r = ReedSolomon::create(3, 2).value();

    {
        auto shards = make_random_shards(10, 4);
        CHECK(r.encode(shards).error() == Error::TooFewShards);
        CHECK(r.verify(shards).error() == Error::TooFewShards);

        auto option_shards = shards_to_option_shards(shards);
        CHECK(r.reconstruct(option_shards).error() == Error::TooFewShards);
    }
    {
        auto shards = make_random_shards(10, 6);
        CHECK(r.encode(shards).error() == Error::TooManyShards);
        CHECK(r.verify(shards).error() == Error::TooManyShards);

        auto option_shards = shards_to_option_shards(shards);
        CHECK(r.reconstruct(option_shards).error() == Error::TooManyShards);
    }
}

// shard 尺寸校验：长度不一致返回 IncorrectShardSize、首个 shard 为空
// 返回 EmptyShard、全部缺失则返回 TooFewShardsPresent。
// 对应 Rust 版的 test_check_slices_or_shards_size。
TEST_CASE("check slices or shards size") {
    const auto r = ReedSolomon::create(2, 2).value();

    {
        Shards shards = {{0, 0, 0}, {0, 1}, {1, 2, 3}, {0, 0, 0}};
        CHECK(r.encode(shards).error() == Error::IncorrectShardSize);
        CHECK(r.verify(shards).error() == Error::IncorrectShardSize);

        auto option_shards = shards_to_option_shards(shards);
        CHECK(r.reconstruct(option_shards).error() == Error::IncorrectShardSize);
    }
    {
        Shards shards = {{0, 1}, {0, 1}, {1, 2, 3}, {0, 0, 0}};
        CHECK(r.encode(shards).error() == Error::IncorrectShardSize);
        CHECK(r.verify(shards).error() == Error::IncorrectShardSize);

        auto option_shards = shards_to_option_shards(shards);
        CHECK(r.reconstruct(option_shards).error() == Error::IncorrectShardSize);
    }
    {
        Shards shards = {{0, 1}, {0, 1, 4}, {1, 2, 3}, {0, 0, 0}};
        CHECK(r.encode(shards).error() == Error::IncorrectShardSize);
        CHECK(r.verify(shards).error() == Error::IncorrectShardSize);

        auto option_shards = shards_to_option_shards(shards);
        CHECK(r.reconstruct(option_shards).error() == Error::IncorrectShardSize);
    }
    {
        Shards shards = {{}, {0, 1, 3}, {1, 2, 3}, {0, 0, 0}};
        CHECK(r.encode(shards).error() == Error::EmptyShard);
        CHECK(r.verify(shards).error() == Error::EmptyShard);

        auto option_shards = shards_to_option_shards(shards);
        CHECK(r.reconstruct(option_shards).error() == Error::EmptyShard);
    }
    {
        OptionShards option_shards(4);
        CHECK(r.reconstruct(option_shards).error() == Error::TooFewShardsPresent);
    }
}

// ShardByShard 逐分片编码：依次喂入每个 data shard 后，
// parity 结果应与一次性 encode 完全一致，期间 cur_input_index 递增。
// 对应 Rust 版的 shardbyshard_encode_correctly。
TEST_CASE("shard by shard encode correctly") {
    const auto r = ReedSolomon::create(10, 3).value();
    ShardByShard sbs(r);

    auto shards = make_random_shards(10'000, 13);
    auto shards_copy = shards;

    REQUIRE(r.encode(shards));

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(sbs.cur_input_index() == i);
        REQUIRE(sbs.encode(shards_copy));
    }

    CHECK(sbs.parity_ready());
    CHECK(shards == shards_copy);

    sbs.reset_force();
    CHECK(sbs.cur_input_index() == 0);
}

// quickcheck 属性：随机参数下 ShardByShard 逐分片编码结果应与 encode
// 相同，并且 reset 后可复用同一个记账器多轮。
// 对应 Rust 版的 qc_shardbyshard_encode_same_as_encode。
TEST_CASE("qc: shard by shard encode same as encode") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params();
        const std::size_t reuse = random_size(10);

        const auto r = ReedSolomon::create(p.data, p.parity).value();
        ShardByShard sbs(r);

        auto expect = make_random_shards(p.size, p.data + p.parity);
        auto shards = expect;

        for (std::size_t round = 0; round < 1 + reuse; ++round) {
            REQUIRE(r.encode(expect));

            for (std::size_t i = 0; i < p.data; ++i) {
                CHECK(sbs.cur_input_index() == i);
                REQUIRE(sbs.encode(shards));
            }

            CHECK(expect == shards);
            CHECK(sbs.parity_ready());
            CHECK(sbs.cur_input_index() == p.data);
            REQUIRE(sbs.reset());
            CHECK(!sbs.parity_ready());
            CHECK(sbs.cur_input_index() == 0);
        }
    }
}

// ShardByShard 的 encode_sep 形态（data 与 parity 分离传入）：
// 结果应与一次性 encode_sep 完全一致。
// 对应 Rust 版的 shardbyshard_encode_sep_correctly。
TEST_CASE("shard by shard encode_sep correctly") {
    const auto r = ReedSolomon::create(10, 3).value();
    ShardByShard sbs(r);

    auto shards = make_random_shards(10'000, 13);
    auto shards_copy = shards;

    const auto data = std::span(shards).first(10);
    const auto parity = std::span(shards).subspan(10);
    const auto data_copy = std::span(shards_copy).first(10);
    const auto parity_copy = std::span(shards_copy).subspan(10);

    REQUIRE(r.encode_sep(data, parity));

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(sbs.cur_input_index() == i);
        REQUIRE(sbs.encode_sep(data_copy, parity_copy));
    }

    CHECK(sbs.parity_ready());
    CHECK(shards == shards_copy);

    sbs.reset_force();
    CHECK(sbs.cur_input_index() == 0);
}

// quickcheck 属性：ShardByShard 的 encode_sep 与一次性 encode_sep
// 结果一致，并验证 reset 复用。
// 对应 Rust 版的 qc_shardbyshard_encode_sep_same_as_encode。
TEST_CASE("qc: shard by shard encode_sep same as encode_sep") {
    for (int iter = 0; iter < QC_ITERS; ++iter) {
        const auto p = random_qc_params();
        const std::size_t reuse = random_size(10);

        const auto r = ReedSolomon::create(p.data, p.parity).value();
        ShardByShard sbs(r);

        auto expect = make_random_shards(p.size, p.data + p.parity);
        auto shards = expect;

        for (std::size_t round = 0; round < 1 + reuse; ++round) {
            {
                const auto data = std::span(expect).first(p.data);
                const auto parity = std::span(expect).subspan(p.data);
                REQUIRE(r.encode_sep(data, parity));
            }
            {
                const auto data = std::span(shards).first(p.data);
                const auto parity = std::span(shards).subspan(p.data);
                for (std::size_t i = 0; i < p.data; ++i) {
                    CHECK(sbs.cur_input_index() == i);
                    REQUIRE(sbs.encode_sep(data, parity));
                }
            }

            CHECK(expect == shards);
            CHECK(sbs.parity_ready());
            CHECK(sbs.cur_input_index() == p.data);
            REQUIRE(sbs.reset());
            CHECK(!sbs.parity_ready());
            CHECK(sbs.cur_input_index() == 0);
        }
    }
}

// 更严格的逐分片编码测试：每次喂入后立即把该 data shard 改写为随机值，
// 证明 ShardByShard 不会依赖已处理过的输入内容。
// 对应 Rust 版的 shardbyshard_encode_correctly_more_rigorous。
TEST_CASE("shard by shard encode correctly (more rigorous)") {
    const auto r = ReedSolomon::create(10, 3).value();
    ShardByShard sbs(r);

    auto shards = make_random_shards(10'000, 13);
    auto shards_copy = make_random_shards(10'000, 13);

    REQUIRE(r.encode(shards));

    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(sbs.cur_input_index() == i);
        shards_copy[i] = shards[i];
        REQUIRE(sbs.encode(shards_copy));
        fill_random(std::span<std::uint8_t>(shards_copy[i]));
    }

    CHECK(sbs.parity_ready());

    for (std::size_t i = 0; i < 10; ++i) {
        shards_copy[i] = shards[i];
    }
    CHECK(shards == shards_copy);

    sbs.reset_force();
    CHECK(sbs.cur_input_index() == 0);
}

// ShardByShard 的错误处理：编码完成后继续调用返回 TooManyCalls；
// 中途 reset 返回 LeftoverShards；空 shard 或尺寸错误经 SBSError
// 包装为 rs_error 返回，且失败不会推进 cur_input_index。
// 对应 Rust 版的 shardbyshard_encode_error_handling。
TEST_CASE("shard by shard encode error handling") {
    {
        const auto r = ReedSolomon::create(10, 3).value();
        ShardByShard sbs(r);

        auto shards = make_random_shards(10'000, 13);

        for (std::size_t i = 0; i < 10; ++i) {
            CHECK(sbs.cur_input_index() == i);
            REQUIRE(sbs.encode(shards));
        }

        CHECK(sbs.parity_ready());
        CHECK(sbs.encode(shards).error() == SBSError::too_many_calls());

        REQUIRE(sbs.reset());

        CHECK(sbs.cur_input_index() == 0);
        REQUIRE(sbs.encode(shards));

        CHECK(sbs.reset().error() == SBSError::leftover_shards());

        sbs.reset_force();
        CHECK(sbs.cur_input_index() == 0);
    }
    {
        const auto r = ReedSolomon::create(10, 3).value();
        ShardByShard sbs(r);

        auto shards = make_random_shards(100, 13);
        shards[0].clear();

        CHECK(sbs.cur_input_index() == 0);
        CHECK(sbs.encode(shards).error() == SBSError::rs_error(Error::EmptyShard));
        CHECK(sbs.cur_input_index() == 0);
        CHECK(sbs.encode(shards).error() == SBSError::rs_error(Error::EmptyShard));
        CHECK(sbs.cur_input_index() == 0);

        shards[0] = std::vector<std::uint8_t>(100, 0);
        REQUIRE(sbs.encode(shards));
        CHECK(sbs.cur_input_index() == 1);
    }
    {
        const auto r = ReedSolomon::create(10, 3).value();
        ShardByShard sbs(r);

        auto shards = make_random_shards(100, 13);
        shards[1] = std::vector<std::uint8_t>(99, 0);

        CHECK(sbs.cur_input_index() == 0);
        CHECK(sbs.encode(shards).error() == SBSError::rs_error(Error::IncorrectShardSize));
        CHECK(sbs.cur_input_index() == 0);
        CHECK(sbs.encode(shards).error() == SBSError::rs_error(Error::IncorrectShardSize));
        CHECK(sbs.cur_input_index() == 0);

        shards[1] = std::vector<std::uint8_t>(100, 0);
        REQUIRE(sbs.encode(shards));
        CHECK(sbs.cur_input_index() == 1);
    }
}

// ShardByShard encode_sep 形态的错误处理（同上，但 data/parity 分离，
// 错误 shard 分别位于 data 半区与 parity 半区）。
// 对应 Rust 版的 shardbyshard_encode_sep_error_handling。
TEST_CASE("shard by shard encode_sep error handling") {
    {
        const auto r = ReedSolomon::create(10, 3).value();
        ShardByShard sbs(r);

        auto shards = make_random_shards(10'000, 13);
        const auto data = std::span(shards).first(10);
        const auto parity = std::span(shards).subspan(10);

        for (std::size_t i = 0; i < 10; ++i) {
            CHECK(sbs.cur_input_index() == i);
            REQUIRE(sbs.encode_sep(data, parity));
        }

        CHECK(sbs.parity_ready());
        CHECK(sbs.encode_sep(data, parity).error() == SBSError::too_many_calls());

        REQUIRE(sbs.reset());

        CHECK(sbs.cur_input_index() == 0);
        REQUIRE(sbs.encode_sep(data, parity));

        CHECK(sbs.reset().error() == SBSError::leftover_shards());

        sbs.reset_force();
        CHECK(sbs.cur_input_index() == 0);
    }
    // 空 shard：分别置于 data 半区和 parity 半区。
    for (const std::size_t bad_index : {std::size_t{0}, std::size_t{10}}) {
        const auto r = ReedSolomon::create(10, 3).value();
        ShardByShard sbs(r);

        auto shards = make_random_shards(100, 13);
        shards[bad_index].clear();
        {
            const auto data = std::span(shards).first(10);
            const auto parity = std::span(shards).subspan(10);

            CHECK(sbs.cur_input_index() == 0);
            CHECK(sbs.encode_sep(data, parity).error() ==
                  SBSError::rs_error(Error::EmptyShard));
            CHECK(sbs.cur_input_index() == 0);
            CHECK(sbs.encode_sep(data, parity).error() ==
                  SBSError::rs_error(Error::EmptyShard));
            CHECK(sbs.cur_input_index() == 0);
        }

        shards[bad_index] = std::vector<std::uint8_t>(100, 0);
        const auto data = std::span(shards).first(10);
        const auto parity = std::span(shards).subspan(10);
        REQUIRE(sbs.encode_sep(data, parity));
        CHECK(sbs.cur_input_index() == 1);
    }
    // 尺寸不正确的 shard：分别置于 data 半区和 parity 半区。
    for (const std::size_t bad_index : {std::size_t{1}, std::size_t{11}}) {
        const auto r = ReedSolomon::create(10, 3).value();
        ShardByShard sbs(r);

        auto shards = make_random_shards(100, 13);
        shards[bad_index] = std::vector<std::uint8_t>(99, 0);
        {
            const auto data = std::span(shards).first(10);
            const auto parity = std::span(shards).subspan(10);

            CHECK(sbs.cur_input_index() == 0);
            CHECK(sbs.encode_sep(data, parity).error() ==
                  SBSError::rs_error(Error::IncorrectShardSize));
            CHECK(sbs.cur_input_index() == 0);
            CHECK(sbs.encode_sep(data, parity).error() ==
                  SBSError::rs_error(Error::IncorrectShardSize));
            CHECK(sbs.cur_input_index() == 0);
        }

        shards[bad_index] = std::vector<std::uint8_t>(100, 0);
        const auto data = std::span(shards).first(10);
        const auto parity = std::span(shards).subspan(10);
        REQUIRE(sbs.encode_sep(data, parity));
        CHECK(sbs.cur_input_index() == 1);
    }
}

// encode_single_sep 成功路径：逐个 data shard 调用后结果应与 encode
// 一致且通过 verify。对应 Rust 版的 test_encode_single_sep。
TEST_CASE("encode_single_sep") {
    const auto r = ReedSolomon::create(10, 3).value();

    auto shards = make_random_shards(10, 13);
    auto shards_copy = shards;

    REQUIRE(r.encode(shards));

    {
        const auto data = std::span(shards_copy).first(10);
        const auto parity = std::span(shards_copy).subspan(10);
        for (std::size_t i = 0; i < 10; ++i) {
            REQUIRE(r.encode_single_sep(i, data[i], parity));
        }
    }
    CHECK(r.verify(shards).value());
    CHECK(r.verify(shards_copy).value());
    check_eq_shards(shards, shards_copy);
}

// encode_sep 成功路径：data 与 parity 分离传入，结果应与 encode 一致。
// 对应 Rust 版的 test_encode_sep。
TEST_CASE("encode_sep") {
    const auto r = ReedSolomon::create(10, 3).value();

    auto shards = make_random_shards(10'000, 13);
    auto shards_copy = shards;

    REQUIRE(r.encode(shards));

    {
        const auto data = std::span(shards_copy).first(10);
        const auto parity = std::span(shards_copy).subspan(10);
        REQUIRE(r.encode_sep(data, parity));
    }

    check_eq_shards(shards, shards_copy);
}

// encode_single_sep 的错误处理：下标越界返回 InvalidIndex，
// data/parity 划分不符合编解码器配置时返回相应的 parity 数量错误。
// 对应 Rust 版的 test_encode_single_sep_error_handling。
TEST_CASE("encode_single_sep error handling") {
    const auto r = ReedSolomon::create(10, 3).value();

    auto shards = make_random_shards(1000, 13);

    {
        const auto data = std::span(shards).first(10);
        const auto parity = std::span(shards).subspan(10);

        for (std::size_t i = 0; i < 10; ++i) {
            REQUIRE(r.encode_single_sep(i, data[i], parity));
        }

        for (const std::size_t i : {10, 11, 12, 13, 14}) {
            CHECK(r.encode_single_sep(i, data[0], parity).error() == Error::InvalidIndex);
        }
    }
    {
        const auto data = std::span(shards).first(11);
        const auto parity = std::span(shards).subspan(11);
        CHECK(r.encode_single_sep(0, data[0], parity).error() == Error::TooFewParityShards);
    }
    {
        const auto data = std::span(shards).first(9);
        const auto parity = std::span(shards).subspan(9);
        CHECK(r.encode_single_sep(0, data[0], parity).error() == Error::TooManyParityShards);
    }
}

// encode_sep 的错误处理：data 或 parity 的数量与编解码器配置不符时
// 返回对应的 TooFew/TooMany 错误。
// 对应 Rust 版的 test_encode_sep_error_handling。
TEST_CASE("encode_sep error handling") {
    const auto r = ReedSolomon::create(10, 3).value();

    {
        auto shards = make_random_shards(1000, 13);
        const auto data = std::span(shards).first(10);
        const auto parity = std::span(shards).subspan(10);
        REQUIRE(r.encode_sep(data, parity));
    }
    {
        auto shards = make_random_shards(1000, 12);
        const auto data = std::span(shards).first(9);
        const auto parity = std::span(shards).subspan(9);
        CHECK(r.encode_sep(data, parity).error() == Error::TooFewDataShards);
    }
    {
        auto shards = make_random_shards(1000, 14);
        const auto data = std::span(shards).first(11);
        const auto parity = std::span(shards).subspan(11);
        CHECK(r.encode_sep(data, parity).error() == Error::TooManyDataShards);
    }
    {
        auto shards = make_random_shards(1000, 12);
        const auto data = std::span(shards).first(10);
        const auto parity = std::span(shards).subspan(10);
        CHECK(r.encode_sep(data, parity).error() == Error::TooFewParityShards);
    }
    {
        auto shards = make_random_shards(1000, 14);
        const auto data = std::span(shards).first(10);
        const auto parity = std::span(shards).subspan(10);
        CHECK(r.encode_sep(data, parity).error() == Error::TooManyParityShards);
    }
}

// encode_single 的错误处理：data shard 下标越界（>= data 数）时
// 返回 InvalidIndex。对应 Rust 版的 test_encode_single_error_handling。
TEST_CASE("encode_single error handling") {
    const auto r = ReedSolomon::create(10, 3).value();

    auto shards = make_random_shards(1000, 13);

    for (std::size_t i = 0; i < 10; ++i) {
        REQUIRE(r.encode_single(i, shards));
    }

    for (const std::size_t i : {10, 11, 12, 13, 14}) {
        CHECK(r.encode_single(i, shards).error() == Error::InvalidIndex);
    }
}
