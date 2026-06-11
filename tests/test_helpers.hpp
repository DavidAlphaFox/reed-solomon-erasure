// test_helpers.hpp - 测试套件共享的辅助函数集合，
// 移植自 Rust crate 中 src/tests/mod.rs 里的辅助函数与宏
// （fill_random、make_random_shards!、shards_to_option_shards、
// option_shards_to_shards 等）。
//
// Rust 版的 quickcheck 属性测试在 C++ 里没有对应框架，
// 这里改用"随机参数 + 固定迭代次数的循环"来复现；
// QcParams / random_qc_params 即为此服务。
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "vendor/doctest.h"

namespace rse::test {

// 全局共享的随机数引擎（惰性初始化，用 random_device 播种）。
// Rust 版各测试直接使用 rand::thread_rng()，这里集中为一个引擎。
inline std::mt19937_64& rng() {
    static std::mt19937_64 engine(std::random_device{}());
    return engine;
}

// 生成一个均匀分布的随机字节。
inline std::uint8_t random_u8() {
    return static_cast<std::uint8_t>(rng()());
}

// 生成 [0, bound) 范围内均匀分布的随机下标/尺寸。
inline std::size_t random_size(std::size_t bound) { // 均匀分布于 [0, bound)
    return std::uniform_int_distribution<std::size_t>(0, bound - 1)(rng());
}

// 用随机字节填充一段缓冲区。
// 对应 Rust 版 src/tests/mod.rs 中的 fill_random（GF(2^8) 元素版本）。
inline void fill_random(std::span<std::uint8_t> arr) {
    for (auto& a : arr) {
        a = random_u8();
    }
}

// fill_random 的 GF(2^16) 重载：每个元素是两个字节（[u8; 2]），
// 对应 Rust 版针对 galois_16 元素类型的随机填充。
inline void fill_random(std::span<std::array<std::uint8_t, 2>> arr) {
    for (auto& a : arr) {
        a = {random_u8(), random_u8()};
    }
}

// 构造 count 个长度均为 per_shard 的随机 shard。
// 对应 Rust 版的 make_random_shards! 宏；Elem 默认为 u8（GF(2^8)），
// GF(2^16) 测试以 std::array<std::uint8_t, 2> 实例化。
template <typename Elem = std::uint8_t>
std::vector<std::vector<Elem>> make_random_shards(std::size_t per_shard, std::size_t count) {
    std::vector<std::vector<Elem>> shards(count, std::vector<Elem>(per_shard));
    for (auto& s : shards) {
        fill_random(std::span<Elem>(s));
    }
    return shards;
}

// 将普通 shard 集合转换为 optional 形式（全部为有值状态），
// 供 reconstruct 的 std::optional 接口使用。
// 对应 Rust 版的 shards_to_option_shards（Vec<Vec<u8>> -> Vec<Option<Vec<u8>>>）。
template <typename Elem>
std::vector<std::optional<std::vector<Elem>>>
shards_to_option_shards(const std::vector<std::vector<Elem>>& shards) {
    std::vector<std::optional<std::vector<Elem>>> result;
    result.reserve(shards.size());
    for (const auto& v : shards) {
        result.emplace_back(v);
    }
    return result;
}

// shards_to_option_shards 的逆操作：把 optional shard 集合还原成普通集合，
// 若有任何缺失（nullopt）的 shard 则测试失败。
// 对应 Rust 版的 option_shards_to_shards（遇到 None 会 panic）。
template <typename Elem>
std::vector<std::vector<Elem>>
option_shards_to_shards(const std::vector<std::optional<std::vector<Elem>>>& shards) {
    std::vector<std::vector<Elem>> result;
    result.reserve(shards.size());
    for (const auto& shard : shards) {
        REQUIRE(shard.has_value());
        result.push_back(*shard);
    }
    return result;
}

// Rust 的 quickcheck 测试每轮会随机抽取四个任意的 usize 参数；
// 这里用每次迭代显式抽取随机值的方式复现同样的输入分布。
struct QcParams {
    std::size_t data;    // data shard 数量
    std::size_t parity;  // parity shard 数量
    std::size_t corrupt; // 被破坏/丢失的 shard 数量，< parity + 1
    std::size_t size;    // 每个 shard 的长度（字节数或元素数）
};

// 生成一组随机的 quickcheck 风格参数（data + parity <= 256，
// 满足 GF(2^8) 编解码器的上限）。
// `max_size` 限制 shard 长度上界；Rust 版测试允许到 1_000_000，
// 但 quickcheck 实际生成的值都很小。这里保持较小的尺寸，
// 让整套属性测试跑得足够快。
inline QcParams random_qc_params(std::size_t max_size = 1024) {
    QcParams p{};
    p.data = 1 + random_size(255);
    p.parity = 1 + random_size(255);
    if (p.data + p.parity > 256) {
        p.parity -= p.data + p.parity - 256;
    }
    p.corrupt = random_size(p.parity + 1);
    p.size = 1 + random_size(max_size);
    return p;
}

// 在 [0, bound) 内生成 count 个互不相同的随机位置，
// 用于随机选择被破坏/丢失的 shard 下标。
// 对应 Rust 版 quickcheck 测试中对 corrupt_pos_s 去重取样的逻辑。
inline std::vector<std::size_t> random_distinct_positions(std::size_t count, std::size_t bound) {
    std::vector<std::size_t> positions;
    positions.reserve(count);
    while (positions.size() < count) {
        const std::size_t pos = random_size(bound);
        if (std::ranges::find(positions, pos) == positions.end()) {
            positions.push_back(pos);
        }
    }
    return positions;
}

} // namespace rse::test
