// bandwidth.cpp - 简单的编码带宽基准测试，
// 对应 Rust 版的 benches/bandwidth.rs。
// 在多种 (data, parity, shard 长度) 组合下测量 encode 的吞吐量
// （MiB/s），并打印运行时分发选中的 SIMD 后端。
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <vector>

#include <rse/rse.hpp>

namespace {

// 生成 count 个长度为 per_shard 的随机 shard。
// 使用固定种子（42），保证各次基准运行的输入一致、结果可复现。
std::vector<std::vector<std::uint8_t>> random_shards(std::size_t per_shard, std::size_t count) {
    std::mt19937_64 rng(42);
    std::vector<std::vector<std::uint8_t>> shards(count, std::vector<std::uint8_t>(per_shard));
    for (auto& s : shards) {
        for (auto& b : s) {
            b = static_cast<std::uint8_t>(rng());
        }
    }
    return shards;
}

// 对给定的 (data, parity, per_shard) 配置测量编码带宽：
// 先做 WARMUP 轮预热，再计时 ROUNDS 轮 encode，
// 按 data 字节总量（与 Rust 版口径一致）折算为 MiB/s 输出。
void bench(std::size_t data, std::size_t parity, std::size_t per_shard) {
    const auto r = rse::galois_8::ReedSolomon::create(data, parity).value();
    auto shards = random_shards(per_shard, data + parity);

    constexpr int WARMUP = 3;
    constexpr int ROUNDS = 50;

    for (int i = 0; i < WARMUP; ++i) {
        (void)r.encode(shards);
    }

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ROUNDS; ++i) {
        (void)r.encode(shards);
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

    const double bytes = static_cast<double>(data) * per_shard * ROUNDS;
    std::printf("encode  %3zu+%-2zu shards x %7zu B: %8.2f MiB/s\n", data, parity, per_shard,
                bytes / elapsed.count() / (1024.0 * 1024.0));
}

} // namespace

int main() {
    // 打印实际选中的 SIMD 后端，便于解读不同机器上的结果差异。
    std::printf("simd backend: %.*s\n", static_cast<int>(rse::gf8::simd_backend().size()),
                rse::gf8::simd_backend().data());
    // 10+3 配置下扫描三种 shard 长度（1 KiB / 64 KiB / 1 MiB）。
    for (const auto per_shard : {1u << 10, 1u << 16, 1u << 20}) {
        bench(10, 3, per_shard);
    }
    // 更大的 50+20 配置，shard 长度 64 KiB。
    bench(50, 20, 1u << 16);
    return 0;
}
