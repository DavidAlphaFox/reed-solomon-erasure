// rse.hpp - 伞头文件（一次性引入整个库）。
//
// 本库提供 Reed-Solomon 纠删码的编码器/解码器。
//
// 注意：纠删码本身不直接检测或纠正错误，它的能力是在冗余度足够的
// 前提下重建"缺失"的数据分片。错误检测需要另行实现（例如校验和），
// 重建时把已损坏的分片标记为缺失即可。
//
// 命名空间速览：
//   rse::galois_8 ::ReedSolomon / ShardByShard   GF(2^8)，最多 256 个分片
//   rse::galois_16::ReedSolomon / ShardByShard   GF(2^16)，最多 65536 个分片
// 这两组别名对应 Rust 版的 galois_8::ReedSolomon 等类型别名。
#pragma once

#include "errors.hpp"       // IWYU pragma: export
#include "field.hpp"        // IWYU pragma: export
#include "galois_16.hpp"    // IWYU pragma: export
#include "galois_8.hpp"     // IWYU pragma: export
#include "matrix.hpp"       // IWYU pragma: export
#include "reed_solomon.hpp" // IWYU pragma: export

namespace rse {

// GF(2^8) 域上的编解码器别名（最常用：分片以字节为元素）。
namespace galois_8 {
using Field = gf8::Field;
using ReedSolomon = rse::ReedSolomon<Field>;
using ShardByShard = rse::ShardByShard<Field>;
} // namespace galois_8

// GF(2^16) 域上的编解码器别名（分片元素为两字节，分片数上限 65536）。
namespace galois_16 {
using Field = gf16::Field;
using ReedSolomon = rse::ReedSolomon<Field>;
using ShardByShard = rse::ShardByShard<Field>;
} // namespace galois_16

} // namespace rse
