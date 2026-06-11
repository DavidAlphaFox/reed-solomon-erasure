// shard_by_shard.hpp - 逐分片编码的记账类。
//
// 从 reed_solomon.hpp 中拆出（对应 Rust 版 core.rs 中的 ShardByShard）：
// 它是建立在 ReedSolomon 之上的便利包装，与编解码核心是单向依赖。
//
// 适用场景：流式数据编码——数据分片不是一次性就绪，希望边收边算
// 摊薄编码开销（例如网络收包，逐包编码比攒齐 N 个包再统一编码均匀）。
// 它替使用者维护"下一个该编码的下标"，避免直接调用 encode_single
// 时顺序出错。
//
// 用法：对每个数据分片依次调用 encode（或 encode_sep）；
// 全部 data 个分片编码完成后 parity_ready() 为 true，校验分片可用；
// 之后调用 reset 复位再编码下一批。
#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "detail/shard_view.hpp"
#include "errors.hpp"
#include "field.hpp"
#include "reed_solomon.hpp"

namespace rse {

template <FieldType F>
class ShardByShard {
public:
    using Elem = typename F::Elem;

    explicit ShardByShard(const ReedSolomon<F>& codec) : codec_(&codec), cur_input_(0) {}

    // 校验分片是否已经就绪（所有数据分片都已编码）。
    [[nodiscard]] bool parity_ready() const noexcept {
        return cur_input_ == codec_->data_shard_count();
    }

    // 复位记账状态。已有分片编码但校验分片尚未就绪时返回
    // SBSError::LeftoverShards（防止半途而废的校验分片被误用）。
    [[nodiscard]] std::expected<void, SBSError> reset() {
        if (cur_input_ > 0 && !parity_ready()) {
            return std::unexpected(SBSError::leftover_shards());
        }
        cur_input_ = 0;
        return {};
    }

    // 无条件复位（跳过检查）。
    void reset_force() noexcept { cur_input_ = 0; }

    // 返回当前待编码的数据分片下标。
    [[nodiscard]] std::size_t cur_input_index() const noexcept { return cur_input_; }

    // 用当前下标对应的数据分片部分地构造校验分片。
    // 所有数据分片都已编码后再调用返回 SBSError::TooManyCalls。
    template <detail::MutShardSeqOf<typename F::Elem> T>
    [[nodiscard]] std::expected<void, SBSError> encode(T&& shards) {
        auto spans = detail::to_mut_spans<Elem>(shards);
        return encode_spans(spans);
    }

    [[nodiscard]] std::expected<void, SBSError>
    encode_spans(std::span<std::span<Elem>> shards) {
        if (auto r = sbs_checks(codec_->check_sbs_all(shards)); !r) {
            return r;
        }
        // 参数检查已全部通过，此处失败属于逻辑错误，直接以 value() 断言。
        codec_->encode_single_spans(cur_input_, shards).value();
        ++cur_input_;
        return {};
    }

    // 同 encode，但数据与校验分片分开传入（数据只读）。
    template <detail::ShardSeqOf<typename F::Elem> T, detail::MutShardSeqOf<typename F::Elem> U>
    [[nodiscard]] std::expected<void, SBSError> encode_sep(const T& data, U&& parity) {
        auto data_spans = detail::to_const_spans<Elem>(data);
        auto parity_spans = detail::to_mut_spans<Elem>(parity);
        return encode_sep_spans(data_spans, parity_spans);
    }

    [[nodiscard]] std::expected<void, SBSError>
    encode_sep_spans(std::span<const std::span<const Elem>> data,
                     std::span<std::span<Elem>> parity) {
        if (auto r = sbs_checks(codec_->check_sbs_sep(data, parity)); !r) {
            return r;
        }
        codec_->encode_single_sep_spans(cur_input_, data[cur_input_], parity).value();
        ++cur_input_;
        return {};
    }

private:
    // 编码前置检查的公共部分：先看是否已编码完，再把编解码器常规检查
    // 的错误包装为 SBSError::RSError。检查失败不改变记账状态，
    // 因此修正参数后可以安全重试。
    // 注意求值顺序：parity_ready 的判断必须在传入的检查结果之前，
    // 所以这里接收的是惰性意义上"已求值"的结果——两处调用点的检查
    // 本身无副作用，先后求值等价。
    [[nodiscard]] std::expected<void, SBSError>
    sbs_checks(std::expected<void, Error> codec_check) const {
        if (parity_ready()) {
            return std::unexpected(SBSError::too_many_calls());
        }
        if (!codec_check) {
            return std::unexpected(SBSError::rs_error(codec_check.error()));
        }
        return {};
    }

    const ReedSolomon<F>* codec_;
    std::size_t cur_input_; // 下一个待编码的数据分片下标
};

} // namespace rse
