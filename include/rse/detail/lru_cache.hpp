// lru_cache.hpp - 极简 LRU 缓存，用于缓存解码矩阵。
//
// Rust 版依赖第三方 crate `lru::LruCache`；C++ 版用
// "双向链表（按使用新旧排序）+ 哈希表（键 → 链表节点）" 自行实现，
// get / put 均摊 O(1)。仅满足本库需要，不是通用容器。
#pragma once

#include <cstddef>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rse::detail {

// vector<size_t> 的哈希函数（标准库没有为 vector 提供 std::hash）。
// 解码矩阵缓存以"缺失分片的下标列表"为键，需要这个哈希。
// 采用 boost::hash_combine 风格的混合方式。
struct IndexVecHash {
    [[nodiscard]] std::size_t operator()(const std::vector<std::size_t>& v) const noexcept {
        std::size_t seed = v.size();
        for (const std::size_t x : v) {
            seed ^= x + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// 固定容量的 LRU（最近最少使用）缓存。
// items_ 链表头部为最近使用的条目；map_ 保存键到链表节点的映射。
template <typename K, typename V, typename Hash = std::hash<K>>
class LruCache {
public:
    explicit LruCache(std::size_t capacity) : capacity_(capacity) {}

    // 查询：命中则把该条目移到链表头部（标记为最近使用）并返回其值副本；
    // 未命中返回 nullopt。
    [[nodiscard]] std::optional<V> get(const K& key) {
        const auto it = map_.find(key);
        if (it == map_.end()) {
            return std::nullopt;
        }
        // splice 把节点摘下接到头部，迭代器保持有效，无需更新 map_。
        items_.splice(items_.begin(), items_, it->second);
        return it->second->second;
    }

    // 插入或更新：已有键则更新值并移到头部；新键则插入头部，
    // 超出容量时淘汰链表尾部（最久未使用）的条目。
    void put(K key, V value) {
        if (const auto it = map_.find(key); it != map_.end()) {
            it->second->second = std::move(value);
            items_.splice(items_.begin(), items_, it->second);
            return;
        }
        items_.emplace_front(std::move(key), std::move(value));
        map_[items_.front().first] = items_.begin();
        if (map_.size() > capacity_) {
            map_.erase(items_.back().first);
            items_.pop_back();
        }
    }

private:
    std::size_t capacity_;
    std::list<std::pair<K, V>> items_; // 头部 = 最近使用
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator, Hash> map_;
};

} // namespace rse::detail
