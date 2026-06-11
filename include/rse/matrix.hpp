// matrix.hpp - 有限域上的矩阵运算。
//
// 移植自 Rust crate 的 src/matrix.rs。
// Reed-Solomon 编码本质上是"数据向量 × 编码矩阵"：本文件提供构造
// Vandermonde 矩阵、矩阵乘法、增广、高斯消元求逆等编解码所需的全部
// 矩阵原语。
//
// 错误处理约定：Rust 版中会 panic 的维度错误（行长不一致、乘法维度
// 不匹配、对非方阵求逆等）在这里抛出 std::invalid_argument；
// 奇异矩阵（业务上可能出现的情况）通过 std::expected 返回。
#pragma once

#include <cstddef>
#include <expected>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "field.hpp"

namespace rse {

// 矩阵运算可恢复的错误（对应 Rust 版 matrix::Error）。
enum class MatrixError {
    SingularMatrix, // 矩阵奇异，无法求逆
};

// 域 F 上的稠密矩阵，按行优先一维存储。
// 对应 Rust 版的 Matrix<F>（Rust 用 SmallVec 内联小矩阵，这里统一用
// std::vector——解码矩阵的构造本就走 LRU 缓存，差异可忽略）。
template <FieldType F>
class Matrix {
public:
    using Elem = typename F::Elem;

    // 构造 rows x cols 的零矩阵。
    Matrix(std::size_t rows, std::size_t cols)
        : row_count_(rows), col_count_(cols), data_(rows * cols, F::zero()) {}

    // 由嵌套初始化列表构造（主要供测试使用，对应 Rust 的 matrix! 宏）。
    static Matrix new_with_data(std::initializer_list<std::initializer_list<Elem>> init) {
        std::vector<std::vector<Elem>> rows;
        rows.reserve(init.size());
        for (const auto& r : init) {
            rows.emplace_back(r);
        }
        return new_with_data(std::move(rows));
    }

    // 由二维 vector 构造；各行长度不一致时抛出异常（Rust 版 panic）。
    static Matrix new_with_data(std::vector<std::vector<Elem>> init) {
        const std::size_t rows = init.size();
        const std::size_t cols = init[0].size();

        for (const auto& r : init) {
            if (r.size() != cols) {
                throw std::invalid_argument("Inconsistent row sizes");
            }
        }

        Matrix m(rows, cols);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                m.at(r, c) = init[r][c];
            }
        }
        return m;
    }

    // size x size 的单位矩阵。
    static Matrix identity(std::size_t size) {
        Matrix result(size, size);
        for (std::size_t i = 0; i < size; ++i) {
            result.at(i, i) = F::one();
        }
        return result;
    }

    // 构造 Vandermonde 矩阵：第 r 行为 [a_r^0, a_r^1, ..., a_r^(cols-1)]。
    // 只要每行的 a_r 互不相同，任取 rows 行都构成可逆方阵——这是
    // Reed-Solomon 可以从任意 data_shards 个分片恢复数据的数学基础。
    static Matrix vandermonde(std::size_t rows, std::size_t cols) {
        Matrix result(rows, cols);
        for (std::size_t r = 0; r < rows; ++r) {
            // a_r 取域中第 r 个元素：取值本身无所谓，唯一即可。
            const Elem r_a = field::nth<F>(r);
            for (std::size_t c = 0; c < cols; ++c) {
                result.at(r, c) = F::exp(r_a, c);
            }
        }
        return result;
    }

    [[nodiscard]] std::size_t col_count() const noexcept { return col_count_; }
    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] Elem get(std::size_t r, std::size_t c) const { return at(r, c); }
    void set(std::size_t r, std::size_t c, Elem val) { at(r, c) = val; }

    bool operator==(const Matrix&) const = default;

    // 矩阵乘法 this × rhs。维度不匹配抛出异常（Rust 版 panic）。
    [[nodiscard]] Matrix multiply(const Matrix& rhs) const {
        if (col_count_ != rhs.row_count_) {
            throw std::invalid_argument(
                "Column count on left is different from row count on right");
        }
        Matrix result(row_count_, rhs.col_count_);
        for (std::size_t r = 0; r < row_count_; ++r) {
            for (std::size_t c = 0; c < rhs.col_count_; ++c) {
                // 在域 F 上做内积：累加用 F::add（即异或）。
                Elem val = F::zero();
                for (std::size_t i = 0; i < col_count_; ++i) {
                    val = F::add(val, F::mul(at(r, i), rhs.at(i, c)));
                }
                result.at(r, c) = val;
            }
        }
        return result;
    }

    // 水平拼接 [this | rhs]，用于构造求逆所需的增广矩阵。
    [[nodiscard]] Matrix augment(const Matrix& rhs) const {
        if (row_count_ != rhs.row_count_) {
            throw std::invalid_argument("Matrices do not have the same row count");
        }
        Matrix result(row_count_, col_count_ + rhs.col_count_);
        for (std::size_t r = 0; r < row_count_; ++r) {
            for (std::size_t c = 0; c < col_count_; ++c) {
                result.at(r, c) = at(r, c);
            }
            for (std::size_t c = 0; c < rhs.col_count_; ++c) {
                result.at(r, col_count_ + c) = rhs.at(r, c);
            }
        }
        return result;
    }

    // 取子矩阵 [rmin, rmax) x [cmin, cmax)。
    [[nodiscard]] Matrix sub_matrix(std::size_t rmin, std::size_t cmin, std::size_t rmax,
                                    std::size_t cmax) const {
        Matrix result(rmax - rmin, cmax - cmin);
        for (std::size_t r = rmin; r < rmax; ++r) {
            for (std::size_t c = cmin; c < cmax; ++c) {
                result.at(r - rmin, c - cmin) = at(r, c);
            }
        }
        return result;
    }

    // 以只读 span 形式返回某一行（编码核心直接以行为系数向量使用）。
    [[nodiscard]] std::span<const Elem> get_row(std::size_t row) const {
        return std::span<const Elem>(data_).subspan(row * col_count_, col_count_);
    }

    // 交换两行（高斯消元的选主元步骤使用）。
    void swap_rows(std::size_t r1, std::size_t r2) {
        if (r1 == r2) {
            return;
        }
        for (std::size_t i = 0; i < col_count_; ++i) {
            std::swap(data_[r1 * col_count_ + i], data_[r2 * col_count_ + i]);
        }
    }

    [[nodiscard]] bool is_square() const noexcept { return row_count_ == col_count_; }

    // 高斯-约当消元：把矩阵就地化为行最简形。
    // 矩阵奇异（某列找不到非零主元）时返回 SingularMatrix。
    std::expected<void, MatrixError> gaussian_elim() {
        for (std::size_t r = 0; r < row_count_; ++r) {
            // 主元为零时，向下寻找非零行交换上来。
            if (at(r, r) == F::zero()) {
                for (std::size_t r_below = r + 1; r_below < row_count_; ++r_below) {
                    if (at(r_below, r) != F::zero()) {
                        swap_rows(r, r_below);
                        break;
                    }
                }
            }
            // 仍然找不到非零主元：矩阵奇异。
            if (at(r, r) == F::zero()) {
                return std::unexpected(MatrixError::SingularMatrix);
            }
            // 把主元缩放到 1。
            if (at(r, r) != F::one()) {
                const Elem scale = F::div(F::one(), at(r, r));
                for (std::size_t c = 0; c < col_count_; ++c) {
                    at(r, c) = F::mul(scale, at(r, c));
                }
            }
            // 用主元行消去下方各行该列的元素。
            // （GF(2^n) 中加减法都是异或，所以"减去倍数"写作 add。）
            for (std::size_t r_below = r + 1; r_below < row_count_; ++r_below) {
                if (at(r_below, r) != F::zero()) {
                    const Elem scale = at(r_below, r);
                    for (std::size_t c = 0; c < col_count_; ++c) {
                        at(r_below, c) = F::add(at(r_below, c), F::mul(scale, at(r, c)));
                    }
                }
            }
        }

        // 回代：清除主对角线上方的元素，得到行最简形。
        for (std::size_t d = 0; d < row_count_; ++d) {
            for (std::size_t r_above = 0; r_above < d; ++r_above) {
                if (at(r_above, d) != F::zero()) {
                    const Elem scale = at(r_above, d);
                    for (std::size_t c = 0; c < col_count_; ++c) {
                        at(r_above, c) = F::add(at(r_above, c), F::mul(scale, at(d, c)));
                    }
                }
            }
        }
        return {};
    }

    // 求逆：对增广矩阵 [this | I] 做高斯-约当消元，右半部分即为逆矩阵。
    // 非方阵抛出异常（Rust 版 panic）；奇异矩阵通过 expected 返回错误。
    [[nodiscard]] std::expected<Matrix, MatrixError> invert() const {
        if (!is_square()) {
            throw std::invalid_argument("Trying to invert a non-square matrix");
        }

        Matrix work = augment(identity(row_count_));
        if (auto res = work.gaussian_elim(); !res) {
            return std::unexpected(res.error());
        }
        return work.sub_matrix(0, row_count_, col_count_, col_count_ * 2);
    }

private:
    // 行优先下标访问（对应 Rust 版的 acc! 宏）。
    [[nodiscard]] Elem& at(std::size_t r, std::size_t c) { return data_[r * col_count_ + c]; }
    [[nodiscard]] const Elem& at(std::size_t r, std::size_t c) const {
        return data_[r * col_count_ + c];
    }

    std::size_t row_count_;
    std::size_t col_count_;
    std::vector<Elem> data_; // 行优先的一维扁平存储
};

} // namespace rse
