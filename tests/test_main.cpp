// test_main.cpp - 测试程序入口。
// 仅负责让 doctest 生成 main 函数；所有测试用例分布在其余 test_*.cpp 中。
// Rust 版无对应文件（cargo test 自带测试入口）。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vendor/doctest.h"
