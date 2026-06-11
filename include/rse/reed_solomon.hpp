// reed_solomon.hpp - Reed-Solomon 纠删码编解码器核心。
//
// 移植自 Rust crate 的 src/core.rs。
//
// 编码原理简述：
//   构造一个 (data+parity) x data 的编码矩阵，其上半部分是 data x data
//   的单位阵（数据分片编码后保持原样，即"系统码"），下半部分的每一行
//   是一组校验系数。编码 = 用校验行与数据分片做域上的矩阵乘法；
//   重建 = 从尚存分片对应的行中取出 data 行组成方阵并求逆，再用逆矩阵
//   的相应行乘以现存分片恢复缺失数据。
//
// 与 Rust 版的 API 对应关系：
//   - Result<T, Error>            -> std::expected<T, Error>
//   - AsRef<[Elem]> 泛型分片序列  -> ShardSeqOf / MutShardSeqOf concept
//     （见 detail/shard_view.hpp）
//   - Option<Vec<u8>> 分片        -> std::optional<std::vector<Elem>>
//   - (&mut [u8], bool) 分片      -> (分片序列, std::span<const bool>) 两参数
//   - ReconstructShard trait      -> 内部的 OptionShardAccess / FlaggedShardAccess
//
// 逐分片编码的记账类 ShardByShard 在 shard_by_shard.hpp。
#pragma once

#include <algorithm>
#include <cstddef>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "detail/lru_cache.hpp"
#include "detail/shard_view.hpp"
#include "errors.hpp"
#include "field.hpp"
#include "matrix.hpp"

namespace rse {

// 解码矩阵 LRU 缓存的容量（与 Rust 版常量一致）。
// 缓存键是"缺失分片的下标列表"——同样的缺失模式重复出现时
// 可以跳过开销最大的矩阵求逆。
inline constexpr std::size_t DATA_DECODE_MATRIX_CACHE_CAPACITY = 254;

// Reed-Solomon 纠删码编解码器。
//
// 接口分组（与 Rust 版一致）：
//   - 编码：encode / encode_sep（数据与校验分开传入，数据只读）
//   - 逐分片编码：encode_single / encode_single_sep
//     （容易误用，推荐通过 ShardByShard 记账类使用）
//   - 校验：verify / verify_with_buffer（后者复用调用方缓冲区）
//   - 重建：reconstruct / reconstruct_data（后者只恢复数据分片）
//
// 通用错误约定：
//   分片数量不符 -> TooFew*/TooMany*；首分片为空 -> EmptyShard；
//   各分片长度不一致 -> IncorrectShardSize；
//   重建时现存分片不足 -> TooFewShardsPresent。
//
// 线程安全：所有方法都是 const 的；内部解码矩阵缓存由互斥锁保护，
// 同一实例可在多线程中并发使用（与 Rust 版相同）。
template <FieldType F>
class ReedSolomon {
public:
    using Elem = typename F::Elem;
    using Shard = std::vector<Elem>;
    using OptionShard = std::optional<Shard>;

    // 创建编解码器实例（对应 Rust 的 ReedSolomon::new）。
    //
    // 返回错误：
    //   data_shards == 0                       -> TooFewDataShards
    //   parity_shards == 0                     -> TooFewParityShards
    //   data_shards + parity_shards > F::ORDER -> TooManyShards
    //     （编码矩阵的行必须对应互不相同的域元素，所以分片总数受域阶限制）
    [[nodiscard]] static std::expected<ReedSolomon, Error> create(std::size_t data_shards,
                                                                  std::size_t parity_shards) {
        if (data_shards == 0) {
            return std::unexpected(Error::TooFewDataShards);
        }
        if (parity_shards == 0) {
            return std::unexpected(Error::TooFewParityShards);
        }
        if (data_shards + parity_shards > F::ORDER) {
            return std::unexpected(Error::TooManyShards);
        }
        return ReedSolomon(data_shards, parity_shards);
    }

    // 拷贝构造：按同样参数重建（对应 Rust 版 Clone 的实现方式）。
    // 解码矩阵缓存不随拷贝转移——它只是性能优化。
    ReedSolomon(const ReedSolomon& other)
        : ReedSolomon(other.data_shard_count_, other.parity_shard_count_) {}

    ReedSolomon& operator=(const ReedSolomon& other) {
        if (this != &other) {
            ReedSolomon tmp(other);
            data_shard_count_ = tmp.data_shard_count_;
            parity_shard_count_ = tmp.parity_shard_count_;
            total_shard_count_ = tmp.total_shard_count_;
            matrix_ = std::move(tmp.matrix_);
            // 缓存只是缓存，赋值后从空开始即可。
            cache_ = LruMatrixCache(DATA_DECODE_MATRIX_CACHE_CAPACITY);
        }
        return *this;
    }

    // 移动构造/赋值：互斥锁不可移动，各自保留自己的锁，
    // 其余成员正常移动。
    ReedSolomon(ReedSolomon&& other) noexcept
        : data_shard_count_(other.data_shard_count_),
          parity_shard_count_(other.parity_shard_count_),
          total_shard_count_(other.total_shard_count_),
          matrix_(std::move(other.matrix_)),
          cache_(std::move(other.cache_)) {}

    ReedSolomon& operator=(ReedSolomon&& other) noexcept {
        if (this != &other) {
            data_shard_count_ = other.data_shard_count_;
            parity_shard_count_ = other.parity_shard_count_;
            total_shard_count_ = other.total_shard_count_;
            matrix_ = std::move(other.matrix_);
            cache_ = std::move(other.cache_);
        }
        return *this;
    }

    // 相等性只看配置参数（对应 Rust 版 PartialEq 的实现）。
    [[nodiscard]] bool operator==(const ReedSolomon& rhs) const noexcept {
        return data_shard_count_ == rhs.data_shard_count_ &&
               parity_shard_count_ == rhs.parity_shard_count_;
    }

    [[nodiscard]] std::size_t data_shard_count() const noexcept { return data_shard_count_; }
    [[nodiscard]] std::size_t parity_shard_count() const noexcept { return parity_shard_count_; }
    [[nodiscard]] std::size_t total_shard_count() const noexcept { return total_shard_count_; }

    // --- 编码 ---

    // 编码：根据前 data_shard_count 个数据分片计算校验分片，
    // 写入后 parity_shard_count 个槽位（无需事先清零，会被完整覆盖）。
    // 模板版本接受任意可写分片序列，摊平成 span 后转发给 *_spans 核心。
    template <detail::MutShardSeqOf<typename F::Elem> T>
    [[nodiscard]] std::expected<void, Error> encode(T&& shards) const {
        auto spans = detail::to_mut_spans<Elem>(shards);
        return encode_spans(spans);
    }

    [[nodiscard]] std::expected<void, Error>
    encode_spans(std::span<std::span<Elem>> slices) const {
        if (auto r = check_count(slices.size(), total_shard_count_, Error::TooFewShards,
                                 Error::TooManyShards);
            !r)
            return r;
        if (auto r = check_slices_multi(slices); !r) return r;

        // 切成输入（数据分片）与输出（校验分片）两段。
        const auto input = slices.first(data_shard_count_);
        const auto output = slices.subspan(data_shard_count_);

        std::vector<std::span<const Elem>> data(input.begin(), input.end());
        return encode_sep_spans(data, output);
    }

    // 编码（分离式）：数据分片以只读视图传入，只写校验分片。
    // 适合在编码的同时让其他线程只读地使用数据分片。
    template <detail::ShardSeqOf<typename F::Elem> T, detail::MutShardSeqOf<typename F::Elem> U>
    [[nodiscard]] std::expected<void, Error> encode_sep(const T& data, U&& parity) const {
        auto data_spans = detail::to_const_spans<Elem>(data);
        auto parity_spans = detail::to_mut_spans<Elem>(parity);
        return encode_sep_spans(data_spans, parity_spans);
    }

    [[nodiscard]] std::expected<void, Error>
    encode_sep_spans(std::span<const std::span<const Elem>> data,
                     std::span<std::span<Elem>> parity) const {
        if (auto r = check_count(data.size(), data_shard_count_, Error::TooFewDataShards,
                                 Error::TooManyDataShards);
            !r)
            return r;
        if (auto r = check_count(parity.size(), parity_shard_count_, Error::TooFewParityShards,
                                 Error::TooManyParityShards);
            !r)
            return r;
        if (auto r = check_slices_multi_multi(data, parity); !r) return r;

        // 用编码矩阵的校验行做实际计算。
        const auto parity_rows = get_parity_rows();
        code_some_slices(parity_rows, data, parity);
        return {};
    }

    // 逐分片编码：只用下标为 i_data 的数据分片"部分地"构造校验分片。
    //
    // 警告：必须严格按 0..data_shard_count 的顺序依次调用，
    // 否则校验分片结果错误。推荐使用 ShardByShard 记账类代替直接调用。
    template <detail::MutShardSeqOf<typename F::Elem> T>
    [[nodiscard]] std::expected<void, Error> encode_single(std::size_t i_data,
                                                           T&& shards) const {
        auto spans = detail::to_mut_spans<Elem>(shards);
        return encode_single_spans(i_data, spans);
    }

    [[nodiscard]] std::expected<void, Error>
    encode_single_spans(std::size_t i_data, std::span<std::span<Elem>> slices) const {
        if (i_data >= data_shard_count_) {
            return std::unexpected(Error::InvalidIndex);
        }
        if (auto r = check_count(slices.size(), total_shard_count_, Error::TooFewShards,
                                 Error::TooManyShards);
            !r)
            return r;
        if (auto r = check_slices_multi(slices); !r) return r;

        const std::span<const Elem> input = slices[i_data];
        const auto output = slices.subspan(data_shard_count_);
        return encode_single_sep_spans(i_data, input, output);
    }

    // 逐分片编码（分离式）：单独传入第 i_data 个数据分片与校验分片组。
    // i_data == 0 的调用会完整覆盖校验分片，因此正确使用时无需预清零。
    template <detail::MutShardSeqOf<typename F::Elem> U>
    [[nodiscard]] std::expected<void, Error>
    encode_single_sep(std::size_t i_data, std::span<const Elem> single_data, U&& parity) const {
        auto parity_spans = detail::to_mut_spans<Elem>(parity);
        return encode_single_sep_spans(i_data, single_data, parity_spans);
    }

    [[nodiscard]] std::expected<void, Error>
    encode_single_sep_spans(std::size_t i_data, std::span<const Elem> single_data,
                            std::span<std::span<Elem>> parity) const {
        if (i_data >= data_shard_count_) {
            return std::unexpected(Error::InvalidIndex);
        }
        if (auto r = check_count(parity.size(), parity_shard_count_, Error::TooFewParityShards,
                                 Error::TooManyParityShards);
            !r)
            return r;
        if (auto r = check_slices_multi(parity); !r) return r;
        if (parity[0].size() != single_data.size()) {
            return std::unexpected(Error::IncorrectShardSize);
        }

        const auto parity_rows = get_parity_rows();
        code_single_slice(parity_rows, i_data, single_data, parity);
        return {};
    }

    // --- 校验 ---

    // 校验所有分片是否一致（校验分片确实是数据分片的正确编码）。
    // 内部分配一块与校验分片同尺寸的临时缓冲区后转发给
    // verify_with_buffer；频繁调用时建议直接用带缓冲区的版本。
    template <detail::ShardSeqOf<typename F::Elem> T>
    [[nodiscard]] std::expected<bool, Error> verify(const T& shards) const {
        auto spans = detail::to_const_spans<Elem>(shards);
        return verify_spans(spans);
    }

    [[nodiscard]] std::expected<bool, Error>
    verify_spans(std::span<const std::span<const Elem>> slices) const {
        if (auto r = check_count(slices.size(), total_shard_count_, Error::TooFewShards,
                                 Error::TooManyShards);
            !r) {
            return std::unexpected(r.error());
        }
        if (auto r = check_slices_multi(slices); !r) {
            return std::unexpected(r.error());
        }

        const std::size_t slice_len = slices[0].size();
        std::vector<Shard> buffer(parity_shard_count_, Shard(slice_len, F::zero()));
        auto buffer_spans = detail::to_mut_spans<Elem>(buffer);
        return verify_with_buffer_spans(slices, buffer_spans);
    }

    // 校验（带缓冲区）：用调用方提供的缓冲区存放重新计算的校验分片，
    // 避免每次调用都做堆分配。只要返回值不是错误，调用结束后缓冲区中
    // 必然是正确的校验分片——无论校验结果是 true 还是 false。
    template <detail::ShardSeqOf<typename F::Elem> T, detail::MutShardSeqOf<typename F::Elem> U>
    [[nodiscard]] std::expected<bool, Error> verify_with_buffer(const T& shards,
                                                                U&& buffer) const {
        auto spans = detail::to_const_spans<Elem>(shards);
        auto buffer_spans = detail::to_mut_spans<Elem>(buffer);
        return verify_with_buffer_spans(spans, buffer_spans);
    }

    [[nodiscard]] std::expected<bool, Error>
    verify_with_buffer_spans(std::span<const std::span<const Elem>> slices,
                             std::span<std::span<Elem>> buffer) const {
        if (auto r = check_count(slices.size(), total_shard_count_, Error::TooFewShards,
                                 Error::TooManyShards);
            !r) {
            return std::unexpected(r.error());
        }
        if (auto r = check_count(buffer.size(), parity_shard_count_, Error::TooFewBufferShards,
                                 Error::TooManyBufferShards);
            !r) {
            return std::unexpected(r.error());
        }
        if (auto r = check_slices_multi_multi(slices, buffer); !r) {
            return std::unexpected(r.error());
        }

        const auto data = slices.first(data_shard_count_);
        const auto to_check = slices.subspan(data_shard_count_);

        // 重新编码一遍，然后与传入的校验分片逐一比较。
        const auto parity_rows = get_parity_rows();
        code_some_slices(parity_rows, data, buffer);

        for (std::size_t i = 0; i < buffer.size(); ++i) {
            if (!std::ranges::equal(buffer[i], to_check[i])) {
                return false;
            }
        }
        return true;
    }

    // --- 重建 ---

    // 从 "optional 分片" 重建所有分片：nullopt 表示缺失，重建后就地
    // 填充为新分配的缓冲区。所有现存分片长度必须一致。
    // 检查不通过返回错误时不修改任何分片。
    [[nodiscard]] std::expected<void, Error> reconstruct(std::span<OptionShard> shards) const {
        OptionShardAccess access{shards};
        return reconstruct_internal(access, shards.size(), false);
    }

    [[nodiscard]] std::expected<void, Error>
    reconstruct(std::vector<OptionShard>& shards) const {
        return reconstruct(std::span<OptionShard>(shards));
    }

    // 同上，但只重建数据分片（缺失的校验分片保持 nullopt）。
    [[nodiscard]] std::expected<void, Error>
    reconstruct_data(std::span<OptionShard> shards) const {
        OptionShardAccess access{shards};
        return reconstruct_internal(access, shards.size(), true);
    }

    [[nodiscard]] std::expected<void, Error>
    reconstruct_data(std::vector<OptionShard>& shards) const {
        return reconstruct_data(std::span<OptionShard>(shards));
    }

    // 就地重建：present[i] 标记 shards[i] 当前是否持有有效数据；
    // 重建结果写入已有缓冲区（长度必须与其他分片一致）。
    // 对应 Rust 版以 (&mut [u8], bool) 元组为分片的形态。
    // 注意 present 标志不会被更新——重建成功后调用方自行知晓全部就绪。
    template <detail::MutShardSeqOf<typename F::Elem> T>
    [[nodiscard]] std::expected<void, Error> reconstruct(T&& shards,
                                                         std::span<const bool> present) const {
        auto spans = detail::to_mut_spans<Elem>(shards);
        return reconstruct_spans(spans, present, false);
    }

    // 就地重建，仅数据分片。
    template <detail::MutShardSeqOf<typename F::Elem> T>
    [[nodiscard]] std::expected<void, Error>
    reconstruct_data(T&& shards, std::span<const bool> present) const {
        auto spans = detail::to_mut_spans<Elem>(shards);
        return reconstruct_spans(spans, present, true);
    }

    [[nodiscard]] std::expected<void, Error>
    reconstruct_spans(std::span<std::span<Elem>> shards, std::span<const bool> present,
                      bool data_only) const {
        // 标志数量与分片数量必须一致（Rust 版由元组结构天然保证，
        // C++ 版分成两个参数后需要显式检查）。
        if (present.size() != shards.size()) {
            return std::unexpected(Error::InvalidShardFlags);
        }
        FlaggedShardAccess access{shards, present};
        return reconstruct_internal(access, shards.size(), data_only);
    }

    // ShardByShard 使用的内部检查；不属于稳定 API。
    [[nodiscard]] std::expected<void, Error>
    check_sbs_all(std::span<std::span<Elem>> slices) const {
        if (auto r = check_count(slices.size(), total_shard_count_, Error::TooFewShards,
                                 Error::TooManyShards);
            !r)
            return r;
        return check_slices_multi(std::span<const std::span<Elem>>(slices));
    }

    [[nodiscard]] std::expected<void, Error>
    check_sbs_sep(std::span<const std::span<const Elem>> data,
                  std::span<std::span<Elem>> parity) const {
        if (auto r = check_count(data.size(), data_shard_count_, Error::TooFewDataShards,
                                 Error::TooManyDataShards);
            !r)
            return r;
        if (auto r = check_count(parity.size(), parity_shard_count_, Error::TooFewParityShards,
                                 Error::TooManyParityShards);
            !r)
            return r;
        return check_slices_multi_multi(data, parity);
    }

private:
    using LruMatrixCache =
        detail::LruCache<std::vector<std::size_t>, std::shared_ptr<const Matrix<F>>,
                         detail::IndexVecHash>;

    ReedSolomon(std::size_t data_shards, std::size_t parity_shards)
        : data_shard_count_(data_shards),
          parity_shard_count_(parity_shards),
          total_shard_count_(data_shards + parity_shards),
          matrix_(build_matrix(data_shards, data_shards + parity_shards)),
          cache_(DATA_DECODE_MATRIX_CACHE_CAPACITY) {}

    // 构造编码矩阵：取 total x data 的 Vandermonde 矩阵 V，
    // 右乘其顶部 data x data 子阵的逆。这样得到的矩阵顶部恰为单位阵
    // （系统码：数据分片原样保留），且任意 data 行依然线性无关。
    static Matrix<F> build_matrix(std::size_t data_shards, std::size_t total_shards) {
        const Matrix<F> vandermonde = Matrix<F>::vandermonde(total_shards, data_shards);
        const Matrix<F> top = vandermonde.sub_matrix(0, 0, data_shards, data_shards);
        // Vandermonde 子阵必然可逆，求逆不会失败。
        return vandermonde.multiply(top.invert().value());
    }

    // --- 参数检查（对应 Rust 版的 check_piece_count! / check_slices! 宏）---

    // 数量必须恰好等于期望值，否则按多/少返回相应错误。
    // 统一了原先 all/data/parity/parity_buf 四个重复的检查函数。
    [[nodiscard]] static std::expected<void, Error>
    check_count(std::size_t n, std::size_t expected, Error too_few, Error too_many) {
        if (n < expected) return std::unexpected(too_few);
        if (n > expected) return std::unexpected(too_many);
        return {};
    }

    // 检查一组分片：首分片非空、所有分片长度一致。
    // 对应 check_slices!(multi => ...)。
    template <typename SpanLike>
    [[nodiscard]] static std::expected<void, Error>
    check_slices_multi(std::span<SpanLike> slices) {
        const std::size_t size = slices[0].size();
        if (size == 0) {
            return std::unexpected(Error::EmptyShard);
        }
        for (const auto& s : slices) {
            if (s.size() != size) {
                return std::unexpected(Error::IncorrectShardSize);
            }
        }
        return {};
    }

    // 检查两组分片：各自满足 multi 检查，且两组首分片长度一致。
    // 对应 check_slices!(multi => ..., multi => ...)。
    template <typename L, typename R>
    [[nodiscard]] static std::expected<void, Error> check_slices_multi_multi(std::span<L> left,
                                                                             std::span<R> right) {
        if (auto r = check_slices_multi(left); !r) return r;
        if (auto r = check_slices_multi(right); !r) return r;
        if (left[0].size() != right[0].size()) {
            return std::unexpected(Error::IncorrectShardSize);
        }
        return {};
    }

    // --- 编码核心 ---

    // 取出编码矩阵中的全部校验行（下标 data..total 的行）。
    [[nodiscard]] std::vector<std::span<const Elem>> get_parity_rows() const {
        std::vector<std::span<const Elem>> parity_rows;
        parity_rows.reserve(parity_shard_count_);
        for (std::size_t i = data_shard_count_; i < total_shard_count_; ++i) {
            parity_rows.push_back(matrix_.get_row(i));
        }
        return parity_rows;
    }

    // 矩阵-向量乘法的核心：outputs[r] = Σ_i matrix_rows[r][i] * inputs[i]。
    // 按输入分片逐个累加，使内层循环始终是对整条切片的批量
    // mul_slice / mul_slice_add 操作（SIMD 友好的访问模式）。
    template <typename In, typename Out>
    void code_some_slices(std::span<const std::span<const Elem>> matrix_rows,
                          std::span<In> inputs, std::span<Out> outputs) const {
        for (std::size_t i_input = 0; i_input < data_shard_count_; ++i_input) {
            code_single_slice(matrix_rows, i_input, inputs[i_input], outputs);
        }
    }

    // 累加单个输入分片对所有输出分片的贡献：
    // i_input == 0 时直接覆盖输出（mul_slice），免去预清零；
    // 之后的输入用乘加（mul_slice_add）累积。
    template <typename Out>
    void code_single_slice(std::span<const std::span<const Elem>> matrix_rows,
                           std::size_t i_input, std::span<const Elem> input,
                           std::span<Out> outputs) const {
        for (std::size_t i_row = 0; i_row < outputs.size(); ++i_row) {
            const Elem matrix_row_to_use = matrix_rows[i_row][i_input];
            std::span<Elem> output = outputs[i_row];
            if (i_input == 0) {
                field::mul_slice<F>(matrix_row_to_use, input, output);
            } else {
                field::mul_slice_add<F>(matrix_row_to_use, input, output);
            }
        }
    }

    // 获取数据解码矩阵（带 LRU 缓存）。
    //
    // valid_indices：现存分片对应的编码矩阵行号（恰好 data 个）；
    // invalid_indices：缺失分片的行号，作为缓存键。
    [[nodiscard]] std::shared_ptr<const Matrix<F>>
    get_data_decode_matrix(std::span<const std::size_t> valid_indices,
                           std::span<const std::size_t> invalid_indices) const {
        std::vector<std::size_t> key(invalid_indices.begin(), invalid_indices.end());
        {
            // 先查缓存（持锁窗口尽量小：求逆在锁外进行）。
            const std::scoped_lock lock(cache_mutex_);
            if (auto entry = cache_.get(key)) {
                return *entry;
            }
        }
        // 从编码矩阵中取出现存分片对应的行，组成 data x data 方阵。
        // 该方阵把原始数据映射到"我们手上现存的分片"。
        Matrix<F> sub_matrix(data_shard_count_, data_shard_count_);
        for (std::size_t sub_matrix_row = 0; sub_matrix_row < valid_indices.size();
             ++sub_matrix_row) {
            const std::size_t valid_index = valid_indices[sub_matrix_row];
            for (std::size_t c = 0; c < data_shard_count_; ++c) {
                sub_matrix.set(sub_matrix_row, c, matrix_.get(valid_index, c));
            }
        }
        // 求逆即得到"从现存分片还原原始数据"的矩阵。
        // 注意它只能直接生成数据分片；校验分片需要在数据齐全后再编码。
        auto data_decode_matrix =
            std::make_shared<const Matrix<F>>(sub_matrix.invert().value());
        {
            const std::scoped_lock lock(cache_mutex_);
            cache_.put(std::move(key), data_decode_matrix);
        }
        return data_decode_matrix;
    }

    // --- 重建核心 ---

    // 分片访问器：把两种重建输入（optional 分片 / 缓冲区+present 标志）
    // 统一成相同的三操作接口，对应 Rust 版的 ReconstructShard trait：
    //   len(i)                  -> 分片现存时返回长度，缺失返回 nullopt
    //   get(i)                  -> 现存时返回可写视图，缺失返回 nullopt
    //   get_or_initialize(i, n) -> 取出或初始化分片，结果见 InitResult
    enum class InitStatus {
        Present,     // 分片本来就存在（对应 Rust 的 Ok(_)）
        Initialized, // 分片缺失，现已初始化好供写入（对应 Err(Ok(_))）
        Failed,      // 初始化失败，携带错误（对应 Err(Err(_))）
    };

    struct InitResult {
        InitStatus status;
        std::span<Elem> data; // status != Failed 时有效
        Error err{};          // status == Failed 时有效
    };

    // optional 分片的访问器：缺失分片在 get_or_initialize 时
    // 新分配一块全零缓冲区。
    struct OptionShardAccess {
        std::span<OptionShard> shards;

        [[nodiscard]] std::optional<std::size_t> len(std::size_t i) const {
            const auto& s = shards[i];
            return s.has_value() ? std::optional(s->size()) : std::nullopt;
        }

        [[nodiscard]] std::optional<std::span<Elem>> get(std::size_t i) {
            auto& s = shards[i];
            if (!s.has_value()) {
                return std::nullopt;
            }
            return std::span<Elem>(*s);
        }

        [[nodiscard]] InitResult get_or_initialize(std::size_t i, std::size_t len) {
            auto& s = shards[i];
            if (s.has_value()) {
                return {InitStatus::Present, std::span<Elem>(*s)};
            }
            s.emplace(len, F::zero());
            return {InitStatus::Initialized, std::span<Elem>(*s)};
        }
    };

    // "缓冲区 + present 标志"的访问器：缓冲区已经存在，只是内容标记
    // 为无效；长度不符时报 IncorrectShardSize（与 Rust 版 (T, bool)
    // 实现一致）。
    struct FlaggedShardAccess {
        std::span<std::span<Elem>> shards;
        std::span<const bool> present;

        [[nodiscard]] std::optional<std::size_t> len(std::size_t i) const {
            return present[i] ? std::optional(shards[i].size()) : std::nullopt;
        }

        [[nodiscard]] std::optional<std::span<Elem>> get(std::size_t i) {
            return present[i] ? std::optional(shards[i]) : std::nullopt;
        }

        [[nodiscard]] InitResult get_or_initialize(std::size_t i, std::size_t len) {
            if (shards[i].size() != len) {
                return {InitStatus::Failed, {}, Error::IncorrectShardSize};
            }
            if (present[i]) {
                return {InitStatus::Present, shards[i]};
            }
            return {InitStatus::Initialized, shards[i]};
        }
    };

    // 重建的公共实现（对应 Rust 版 reconstruct_internal）。
    // data_only 为 true 时只恢复数据分片，跳过校验分片的重算。
    template <typename Access>
    std::expected<void, Error> reconstruct_internal(Access& shards, std::size_t shard_count,
                                                    bool data_only) const {
        if (auto r = check_count(shard_count, total_shard_count_, Error::TooFewShards,
                                 Error::TooManyShards);
            !r)
            return r;

        // 第一遍扫描：统计现存分片数并核对长度一致性。
        // 如果所有分片都在，直接成功返回，什么都不用做。
        std::size_t number_present = 0;
        std::optional<std::size_t> shard_len;

        for (std::size_t i = 0; i < shard_count; ++i) {
            if (const auto len = shards.len(i)) {
                if (*len == 0) {
                    return std::unexpected(Error::EmptyShard);
                }
                ++number_present;
                if (shard_len && *len != *shard_len) {
                    return std::unexpected(Error::IncorrectShardSize);
                }
                shard_len = *len;
            }
        }

        if (number_present == total_shard_count_) {
            return {};
        }

        // 现存分片不足 data 个：信息量不够，无法重建。
        if (number_present < data_shard_count_) {
            return std::unexpected(Error::TooFewShardsPresent);
        }

        // 第二遍扫描：把分片分组——
        //   sub_shards          : 选作解码输入的现存分片（恰好 data 个，
        //                         解码矩阵是 data x data 方阵，多了用不上）
        //   missing_data_slices : 待重建的数据分片输出缓冲区
        //   missing_parity_slices : 待重建的校验分片输出缓冲区
        //   valid_indices       : sub_shards 对应的矩阵行号
        //   invalid_indices     : 缺失分片的行号（兼作解码矩阵缓存键）
        std::vector<std::span<const Elem>> sub_shards;
        std::vector<std::span<Elem>> missing_data_slices;
        std::vector<std::span<Elem>> missing_parity_slices;
        std::vector<std::size_t> valid_indices;
        std::vector<std::size_t> invalid_indices;
        sub_shards.reserve(data_shard_count_);
        valid_indices.reserve(data_shard_count_);
        invalid_indices.reserve(parity_shard_count_);

        for (std::size_t matrix_row = 0; matrix_row < shard_count; ++matrix_row) {
            // data_only 模式下不为缺失的校验分片做初始化，
            // 只把它记入缺失列表（保持缓存键的一致性）。
            if (matrix_row >= data_shard_count_ && data_only) {
                if (auto shard = shards.get(matrix_row)) {
                    if (sub_shards.size() < data_shard_count_) {
                        sub_shards.push_back(*shard);
                        valid_indices.push_back(matrix_row);
                    }
                } else {
                    invalid_indices.push_back(matrix_row);
                }
                continue;
            }

            const InitResult res = shards.get_or_initialize(matrix_row, *shard_len);
            switch (res.status) {
                case InitStatus::Present:
                    // 已有足够的解码输入后，多余的现存分片直接忽略。
                    if (sub_shards.size() < data_shard_count_) {
                        sub_shards.push_back(res.data);
                        valid_indices.push_back(matrix_row);
                    }
                    break;
                case InitStatus::Initialized:
                    if (matrix_row < data_shard_count_) {
                        missing_data_slices.push_back(res.data);
                    } else {
                        missing_parity_slices.push_back(res.data);
                    }
                    invalid_indices.push_back(matrix_row);
                    break;
                case InitStatus::Failed:
                    return std::unexpected(res.err);
            }
        }

        const auto data_decode_matrix =
            get_data_decode_matrix(valid_indices, invalid_indices);

        // 第一步：重建缺失的数据分片。
        // 输入是现存分片（sub_shards），输出是缺失的数据分片，
        // 系数取解码矩阵中对应缺失行的行向量。
        std::vector<std::span<const Elem>> matrix_rows;
        matrix_rows.reserve(parity_shard_count_);
        for (const std::size_t i_slice : invalid_indices) {
            if (i_slice >= data_shard_count_) {
                break; // invalid_indices 升序，遇到校验区即可停止
            }
            matrix_rows.push_back(data_decode_matrix->get_row(i_slice));
        }

        code_some_slices(std::span<const std::span<const Elem>>(matrix_rows),
                         std::span<const std::span<const Elem>>(sub_shards),
                         std::span<std::span<Elem>>(missing_data_slices));

        if (data_only) {
            return {};
        }

        // 第二步：数据分片现已齐全，重算缺失的校验分片。
        // 输入是全部数据分片（含刚重建的），系数取编码矩阵的对应校验行。
        std::vector<std::span<const Elem>> parity_matrix_rows;
        parity_matrix_rows.reserve(parity_shard_count_);
        const auto parity_rows = get_parity_rows();

        for (const std::size_t i_slice : invalid_indices) {
            if (i_slice < data_shard_count_) {
                continue; // 跳过数据区的缺失项
            }
            parity_matrix_rows.push_back(parity_rows[i_slice - data_shard_count_]);
        }

        // 按原始顺序拼出完整的数据分片列表：
        // 未缺失的来自 sub_shards，缺失后重建的来自 missing_data_slices。
        // 两个列表各自有序，按 invalid_indices 中的缺失位置归并。
        std::vector<std::span<const Elem>> all_data_slices;
        all_data_slices.reserve(data_shard_count_);
        {
            std::size_t i_old_data_slice = 0; // sub_shards 的游标
            std::size_t i_new_data_slice = 0; // missing_data_slices 的游标
            std::size_t next_maybe_good = 0;

            // 把 [next_maybe_good, up_to) 范围内的"完好"分片依次推入。
            const auto push_good_up_to = [&](std::size_t up_to) {
                for (std::size_t i = next_maybe_good; i < up_to; ++i) {
                    all_data_slices.push_back(sub_shards[i_old_data_slice]);
                    ++i_old_data_slice;
                }
                next_maybe_good = up_to + 1;
            };

            for (const std::size_t i_slice : invalid_indices) {
                if (i_slice >= data_shard_count_) {
                    break;
                }
                push_good_up_to(i_slice);
                all_data_slices.push_back(missing_data_slices[i_new_data_slice]);
                ++i_new_data_slice;
            }
            push_good_up_to(data_shard_count_);
        }

        // 用完整的数据分片计算缺失的校验分片。
        code_some_slices(std::span<const std::span<const Elem>>(parity_matrix_rows),
                         std::span<const std::span<const Elem>>(all_data_slices),
                         std::span<std::span<Elem>>(missing_parity_slices));

        return {};
    }

    std::size_t data_shard_count_;
    std::size_t parity_shard_count_;
    std::size_t total_shard_count_;
    Matrix<F> matrix_;               // 编码矩阵（顶部单位阵 + 校验行）
    mutable std::mutex cache_mutex_; // 保护下面的解码矩阵缓存
    mutable LruMatrixCache cache_;
};

} // namespace rse
