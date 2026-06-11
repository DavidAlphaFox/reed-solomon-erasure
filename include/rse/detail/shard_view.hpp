// shard_view.hpp - "分片序列"的 concept 与 span 摊平工具（内部实现细节）。
//
// 从 reed_solomon.hpp 中拆出：这组工具是 ReedSolomon 与 ShardByShard
// 共享的输入适配层，与编解码逻辑本身无关。
//
// 这是 Rust 版 `T: AsRef<[U]>, U: AsRef<[Elem]>` 泛型约束和
// convert_2D_slices! 宏的 C++ 对应物。
#pragma once

#include <concepts>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

namespace rse::detail {

// "分片序列"concept：一个 range，其每个元素都是 Elem 的连续 range
// （vector<vector<Elem>>、span 切片、内建数组等均满足）。
template <typename R, typename Elem>
concept ShardSeqOf =
    std::ranges::sized_range<R> && std::ranges::forward_range<R> &&
    std::ranges::contiguous_range<std::ranges::range_reference_t<R>> &&
    std::same_as<std::ranges::range_value_t<std::ranges::range_reference_t<R>>, Elem>;

// 同上，但要求分片内容可写（对应 Rust 的 AsMut<[Elem]>）。
template <typename R, typename Elem>
concept MutShardSeqOf =
    ShardSeqOf<R, Elem> &&
    requires(std::ranges::range_reference_t<R> shard) {
        { std::ranges::data(shard) } -> std::same_as<Elem*>;
    };

// 把任意分片序列摊平成 span 列表。
// Elem 带 const 限定则产出只读视图，否则产出可写视图——
// 一份实现同时覆盖原先的 to_const_spans / to_mut_spans 两个函数。
template <typename Elem, typename R>
    requires ShardSeqOf<R, std::remove_const_t<Elem>>
std::vector<std::span<Elem>> to_spans(R&& shards) {
    std::vector<std::span<Elem>> result;
    result.reserve(std::ranges::size(shards));
    for (auto&& shard : shards) {
        result.emplace_back(std::ranges::data(shard), std::ranges::size(shard));
    }
    return result;
}

// 语义化别名：只读视图。
template <typename Elem, ShardSeqOf<Elem> R>
std::vector<std::span<const Elem>> to_const_spans(R&& shards) {
    return to_spans<const Elem>(std::forward<R>(shards));
}

// 语义化别名：可写视图。
template <typename Elem, MutShardSeqOf<Elem> R>
std::vector<std::span<Elem>> to_mut_spans(R&& shards) {
    return to_spans<Elem>(std::forward<R>(shards));
}

} // namespace rse::detail
