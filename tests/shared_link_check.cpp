// shared_link_check.cpp - 冒烟测试：通过*共享库*（而非静态库）链接，
// 走一遍 encode / verify / reconstruct 完整流程，确认动态库导出的
// 符号齐全且可正常工作。C++ 版特有，Rust 版无对应文件。
#include <cstdio>
#include <optional>
#include <vector>

#include <rse/rse.hpp>

int main() {
    // 4 个 data shard + 2 个 parity shard 的最小编解码器。
    const auto r = rse::galois_8::ReedSolomon::create(4, 2).value();

    std::vector<std::vector<std::uint8_t>> shards = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {0, 0, 0}, {0, 0, 0},
    };
    if (!r.encode(shards)) {
        std::puts("encode failed");
        return 1;
    }
    if (!r.verify(shards).value()) {
        std::puts("verify failed");
        return 1;
    }

    // 丢弃一个 data shard 和一个 parity shard，再用 optional 形态重建。
    std::vector<std::optional<std::vector<std::uint8_t>>> opt(shards.begin(), shards.end());
    opt[0] = std::nullopt;
    opt[5] = std::nullopt;
    if (!r.reconstruct(opt)) {
        std::puts("reconstruct failed");
        return 1;
    }
    if (opt[0].value() != shards[0]) {
        std::puts("reconstructed shard mismatch");
        return 1;
    }

    // 顺带打印运行时分发选中的 SIMD 后端名称。
    std::printf("ok (simd backend: %.*s)\n",
                static_cast<int>(rse::gf8::simd_backend().size()),
                rse::gf8::simd_backend().data());
    return 0;
}
