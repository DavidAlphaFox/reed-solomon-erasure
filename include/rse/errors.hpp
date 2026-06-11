// errors.hpp - Reed-Solomon 纠删码的错误类型定义。
//
// 移植自 Rust crate `reed-solomon-erasure` 的 src/errors.rs。
// Rust 版通过 Result<T, Error> 返回错误；C++ 版与之对应，所有可失败的
// 接口返回 std::expected<T, rse::Error>。
#pragma once

#include <string_view>
#include <variant>

namespace rse {

// 编解码操作可能返回的错误（与 Rust 版的 `Error` 枚举一一对应）。
//
// 命名约定："Shards" 指传入的分片总数，"DataShards"/"ParityShards" 指
// 数据/校验分片数，"BufferShards" 指 verify_with_buffer 的缓冲区分片数。
enum class Error {
    TooFewShards,        // 提供的分片总数少于编解码器配置
    TooManyShards,       // 提供的分片总数多于编解码器配置
    TooFewDataShards,    // 提供的数据分片数少于配置
    TooManyDataShards,   // 提供的数据分片数多于配置
    TooFewParityShards,  // 提供的校验分片数少于配置
    TooManyParityShards, // 提供的校验分片数多于配置
    TooFewBufferShards,  // 缓冲区分片数少于校验分片数
    TooManyBufferShards, // 缓冲区分片数多于校验分片数
    IncorrectShardSize,  // 至少一个分片的长度与其他分片不一致
    TooFewShardsPresent, // 现存分片数不足以重建缺失分片
    EmptyShard,          // 第一个分片长度为零
    InvalidShardFlags,   // present 标志的数量与分片总数不匹配
    InvalidIndex,        // 数据分片下标越界（>= 数据分片数）
};

// 返回错误的英文描述文本。
// 文本内容与 Rust 版 Error::to_string() 逐字相同（测试依赖这一点）。
[[nodiscard]] constexpr std::string_view to_string(Error e) noexcept {
    switch (e) {
        case Error::TooFewShards:
            return "The number of provided shards is smaller than the one in codec";
        case Error::TooManyShards:
            return "The number of provided shards is greater than the one in codec";
        case Error::TooFewDataShards:
            return "The number of provided data shards is smaller than the one in codec";
        case Error::TooManyDataShards:
            return "The number of provided data shards is greater than the one in codec";
        case Error::TooFewParityShards:
            return "The number of provided parity shards is smaller than the one in codec";
        case Error::TooManyParityShards:
            return "The number of provided parity shards is greater than the one in codec";
        case Error::TooFewBufferShards:
            return "The number of provided buffer shards is smaller than the number of parity "
                   "shards in codec";
        case Error::TooManyBufferShards:
            return "The number of provided buffer shards is greater than the number of parity "
                   "shards in codec";
        case Error::IncorrectShardSize:
            return "At least one of the provided shards is not of the correct size";
        case Error::TooFewShardsPresent:
            return "The number of shards present is smaller than number of parity shards, cannot "
                   "reconstruct missing shards";
        case Error::EmptyShard:
            return "The first shard provided is of zero length";
        case Error::InvalidShardFlags:
            return "The number of flags does not match the total number of shards";
        case Error::InvalidIndex:
            return "The data shard index provided is greater or equal to the number of data "
                   "shards in codec";
    }
    return "Unknown error";
}

// 逐分片编码（ShardByShard）的错误类型，对应 Rust 版的 `SBSError`。
//
// Rust 的 SBSError 是带数据的枚举（RSError 变体携带一个 Error）；
// C++ 版用 "Kind 标签 + 可选的 Error 载荷" 来表达同样的结构，
// 并提供与 Rust 各变体对应的静态工厂函数。
class SBSError {
public:
    enum class Kind {
        TooManyCalls,   // 所有数据分片都已编码完，又多调了一次 encode
        LeftoverShards, // 有分片已编码但校验分片尚未就绪时调用了 reset
        RSError,        // 底层编解码检查失败，携带具体的 rse::Error
    };

    // 构造 TooManyCalls 变体。
    static constexpr SBSError too_many_calls() noexcept { return SBSError(Kind::TooManyCalls); }
    // 构造 LeftoverShards 变体。
    static constexpr SBSError leftover_shards() noexcept { return SBSError(Kind::LeftoverShards); }
    // 构造携带底层错误的 RSError 变体。
    static constexpr SBSError rs_error(Error e) noexcept { return SBSError(e); }

    [[nodiscard]] constexpr Kind kind() const noexcept { return kind_; }
    // 取出携带的底层错误；仅当 kind() == Kind::RSError 时有意义。
    [[nodiscard]] constexpr Error rs() const noexcept { return rs_; }

    // 相等比较：Kind 相同，且若为 RSError 则载荷也要相同。
    constexpr bool operator==(const SBSError& other) const noexcept {
        return kind_ == other.kind_ && (kind_ != Kind::RSError || rs_ == other.rs_);
    }

private:
    constexpr explicit SBSError(Kind k) noexcept : kind_(k) {}
    constexpr explicit SBSError(Error e) noexcept : kind_(Kind::RSError), rs_(e) {}

    Kind kind_;
    Error rs_{};
};

// 返回 SBSError 的英文描述文本（与 Rust 版 SBSError::to_string() 相同）。
[[nodiscard]] constexpr std::string_view to_string(const SBSError& e) noexcept {
    switch (e.kind()) {
        case SBSError::Kind::TooManyCalls:
            return "Too many calls";
        case SBSError::Kind::LeftoverShards:
            return "Leftover shards";
        case SBSError::Kind::RSError:
            return to_string(e.rs());
    }
    return "Unknown error";
}

} // namespace rse
