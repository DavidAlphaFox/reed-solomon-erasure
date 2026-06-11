// test_errors.cpp - 错误类型测试，移植自 Rust 版 src/errors.rs 的 tests 模块。
// 验证 Error / SBSError 的字符串描述与 Rust 版逐字一致，
// 并检查 SBSError 的相等性比较（C++ 版补充）。
//
// 在 C++ 移植版中，库函数通过 std::expected<T, rse::Error> 报告错误；
// SBSError 是 ShardByShard 逐分片编码器专用的错误类型。
#include "vendor/doctest.h"

#include <rse/errors.hpp>

using rse::Error;
using rse::SBSError;

// 验证每个 Error 枚举值的 to_string 文本与 Rust 版完全一致。
// 对应 Rust 版 errors.rs 中的 test_error_to_string_is_okay。
TEST_CASE("error to_string is okay") {
    CHECK(rse::to_string(Error::TooFewShards) ==
          "The number of provided shards is smaller than the one in codec");
    CHECK(rse::to_string(Error::TooManyShards) ==
          "The number of provided shards is greater than the one in codec");
    CHECK(rse::to_string(Error::TooFewDataShards) ==
          "The number of provided data shards is smaller than the one in codec");
    CHECK(rse::to_string(Error::TooManyDataShards) ==
          "The number of provided data shards is greater than the one in codec");
    CHECK(rse::to_string(Error::TooFewParityShards) ==
          "The number of provided parity shards is smaller than the one in codec");
    CHECK(rse::to_string(Error::TooManyParityShards) ==
          "The number of provided parity shards is greater than the one in codec");
    CHECK(rse::to_string(Error::TooFewBufferShards) ==
          "The number of provided buffer shards is smaller than the number of parity shards in "
          "codec");
    CHECK(rse::to_string(Error::TooManyBufferShards) ==
          "The number of provided buffer shards is greater than the number of parity shards in "
          "codec");
    CHECK(rse::to_string(Error::IncorrectShardSize) ==
          "At least one of the provided shards is not of the correct size");
    CHECK(rse::to_string(Error::TooFewShardsPresent) ==
          "The number of shards present is smaller than number of parity shards, cannot "
          "reconstruct missing shards");
    CHECK(rse::to_string(Error::EmptyShard) == "The first shard provided is of zero length");
    CHECK(rse::to_string(Error::InvalidShardFlags) ==
          "The number of flags does not match the total number of shards");
    CHECK(rse::to_string(Error::InvalidIndex) ==
          "The data shard index provided is greater or equal to the number of data shards in "
          "codec");
}

// 验证 SBSError 三种形态（TooManyCalls / LeftoverShards / RSError 包装）
// 的 to_string 文本。对应 Rust 版 errors.rs 中的 test_sbs_error_to_string_is_okay。
TEST_CASE("sbs error to_string is okay") {
    CHECK(rse::to_string(SBSError::too_many_calls()) == "Too many calls");
    CHECK(rse::to_string(SBSError::leftover_shards()) == "Leftover shards");
    CHECK(rse::to_string(SBSError::rs_error(Error::TooFewShards)) ==
          rse::to_string(Error::TooFewShards));
}

// 验证 SBSError 的相等/不等比较语义（包括内嵌 Error 的比较）。
// C++ 版新增的测试：Rust 版通过 derive(PartialEq) 自动获得该语义，
// C++ 版需手写 operator==，因此显式加以验证。
TEST_CASE("sbs error equality") {
    CHECK(SBSError::too_many_calls() == SBSError::too_many_calls());
    CHECK(SBSError::rs_error(Error::EmptyShard) == SBSError::rs_error(Error::EmptyShard));
    CHECK(SBSError::rs_error(Error::EmptyShard) != SBSError::rs_error(Error::TooFewShards));
    CHECK(SBSError::too_many_calls() != SBSError::leftover_shards());
}
