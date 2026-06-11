// test_matrix.cpp - GF(2^8) 上矩阵运算的测试，
// 移植自 Rust 版 src/matrix.rs 的 tests 模块。
//
// 覆盖矩阵的行列数查询、行交换、乘法、增广、单位阵与求逆
// （含奇异矩阵和非方阵的错误路径）。Rust 版以 panic 表达的
// 失败用例，在 C++ 版对应抛出 std::invalid_argument；
// 奇异矩阵则通过 std::expected 的错误值返回。
#include "vendor/doctest.h"

#include <stdexcept>

#include <rse/galois_8.hpp>
#include <rse/matrix.hpp>

using Matrix = rse::Matrix<rse::gf8::Field>;

// 验证 col_count 返回正确的列数。对应 Rust 版的 test_matrix_col_count。
TEST_CASE("matrix col count") {
    const auto m1 = Matrix::new_with_data({{1, 0, 0}});
    const auto m2 = Matrix::new_with_data({{0, 0, 0}, {0, 0, 0}});
    const Matrix m3(1, 4);

    CHECK(m1.col_count() == 3);
    CHECK(m2.col_count() == 3);
    CHECK(m3.col_count() == 4);
}

// 验证 row_count 返回正确的行数。对应 Rust 版的 test_matrix_row_count。
TEST_CASE("matrix row count") {
    const auto m1 = Matrix::new_with_data({{1, 0, 0}});
    const auto m2 = Matrix::new_with_data({{0, 0, 0}, {0, 0, 0}});
    const Matrix m3(1, 4);

    CHECK(m1.row_count() == 1);
    CHECK(m2.row_count() == 2);
    CHECK(m3.row_count() == 1);
}

// 验证 swap_rows：交换两行得到期望结果，自身交换则保持不变。
// 对应 Rust 版的 test_matrix_swap_rows。
TEST_CASE("matrix swap rows") {
    {
        auto m1 = Matrix::new_with_data({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        const auto expect = Matrix::new_with_data({{7, 8, 9}, {4, 5, 6}, {1, 2, 3}});
        m1.swap_rows(0, 2);
        CHECK(m1 == expect);
    }
    {
        auto m1 = Matrix::new_with_data({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        const auto expect = m1;
        m1.swap_rows(0, 0);
        CHECK(m1 == expect);
        m1.swap_rows(1, 1);
        CHECK(m1 == expect);
        m1.swap_rows(2, 2);
        CHECK(m1 == expect);
    }
}

// 各行长度不一致时构造矩阵应抛出异常。
// 对应 Rust 版的 test_inconsistent_row_sizes（#[should_panic]）。
TEST_CASE("inconsistent row sizes throw") {
    CHECK_THROWS_AS(Matrix::new_with_data({{1, 0, 0}, {0, 1}, {0, 0, 1}}),
                    std::invalid_argument);
}

// 维度不匹配的矩阵相乘应抛出异常。
// 对应 Rust 版的 test_incompatible_multiply（#[should_panic]）。
TEST_CASE("incompatible multiply throws") {
    const auto m1 = Matrix::new_with_data({{0, 1}, {0, 1}, {0, 1}});
    const auto m2 = Matrix::new_with_data({{0, 1, 2}});

    CHECK_THROWS_AS((void)m1.multiply(m2), std::invalid_argument);
}

// 行数不匹配的矩阵增广（augment）应抛出异常。
// 对应 Rust 版的 test_incompatible_augment（#[should_panic]）。
TEST_CASE("incompatible augment throws") {
    const auto m1 = Matrix::new_with_data({{0, 1}});
    const auto m2 = Matrix::new_with_data({{0, 1}, {2, 3}});

    CHECK_THROWS_AS((void)m1.augment(m2), std::invalid_argument);
}

// 验证 identity 生成正确的单位阵。对应 Rust 版的 test_matrix_identity。
TEST_CASE("matrix identity") {
    const auto m1 = Matrix::identity(3);
    const auto m2 = Matrix::new_with_data({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    CHECK(m1 == m2);
}

// 验证 GF(2^8) 上的矩阵乘法结果与已知期望值一致。
// 对应 Rust 版的 test_matrix_multiply。
TEST_CASE("matrix multiply") {
    const auto m1 = Matrix::new_with_data({{1, 2}, {3, 4}});
    const auto m2 = Matrix::new_with_data({{5, 6}, {7, 8}});
    const auto actual = m1.multiply(m2);
    const auto expect = Matrix::new_with_data({{11, 22}, {19, 42}});
    CHECK(actual == expect);
}

// 可逆矩阵求逆的成功用例：3x3 与 5x5 两个已知矩阵，
// 验证逆矩阵与预先计算好的期望值一致。
// 对应 Rust 版的 test_matrix_inverse_pass_cases。
TEST_CASE("matrix inverse pass cases") {
    {
        const auto m =
            Matrix::new_with_data({{56, 23, 98}, {3, 100, 200}, {45, 201, 123}}).invert();
        REQUIRE(m.has_value());
        const auto expect =
            Matrix::new_with_data({{175, 133, 33}, {130, 13, 245}, {112, 35, 126}});
        CHECK(*m == expect);
    }
    {
        const auto m = Matrix::new_with_data({{1, 0, 0, 0, 0},
                                              {0, 1, 0, 0, 0},
                                              {0, 0, 0, 1, 0},
                                              {0, 0, 0, 0, 1},
                                              {7, 7, 6, 6, 1}})
                           .invert();
        REQUIRE(m.has_value());
        const auto expect = Matrix::new_with_data({{1, 0, 0, 0, 0},
                                                   {0, 1, 0, 0, 0},
                                                   {123, 123, 1, 122, 122},
                                                   {0, 0, 1, 0, 0},
                                                   {0, 0, 0, 1, 0}});
        CHECK(*m == expect);
    }
}

// 非方阵求逆应抛出异常。
// 对应 Rust 版的 test_matrix_inverse_non_square（#[should_panic]）。
TEST_CASE("matrix inverse non-square throws") {
    const auto m = Matrix::new_with_data({{56, 23}, {3, 100}, {45, 201}});
    CHECK_THROWS_AS((void)m.invert(), std::invalid_argument);
}

// 奇异矩阵求逆应返回 SingularMatrix 错误（经由 std::expected）。
// 对应 Rust 版的 test_matrix_inverse_singular。
TEST_CASE("matrix inverse singular") {
    const auto m = Matrix::new_with_data({{4, 2}, {12, 6}});
    const auto res = m.invert();
    REQUIRE(!res.has_value());
    CHECK(res.error() == rse::MatrixError::SingularMatrix);
}
