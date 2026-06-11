// config.hpp - 库级公共配置：版本号、符号导出、目标架构检测。
//
// 本文件集中了所有"只能用预处理器宏表达"的配置，避免它们散落在
// 各个源文件中：
//   - 版本号宏：唯一权威来源，CMakeLists.txt 用文本解析读取它，
//     保证头文件与构建系统的版本号永不漂移；
//   - RSE_API：动态库符号导出/导入控制；
//   - RSE_ARCH_*：目标架构检测，统一各 SIMD 源文件中重复的
//     defined(...) 组合判断。
#pragma once

// --- 版本号（与原 Rust crate 版本保持一致）---
// CMakeLists.txt 通过 file(STRINGS) 解析下面三行，请保持格式不变。
#define RSE_VERSION_MAJOR 6
#define RSE_VERSION_MINOR 0
#define RSE_VERSION_PATCH 0

#define RSE_VERSION_STRING_IMPL2(a, b, c) #a "." #b "." #c
#define RSE_VERSION_STRING_IMPL(a, b, c) RSE_VERSION_STRING_IMPL2(a, b, c)
#define RSE_VERSION_STRING \
    RSE_VERSION_STRING_IMPL(RSE_VERSION_MAJOR, RSE_VERSION_MINOR, RSE_VERSION_PATCH)

// --- 目标架构检测 ---
// 统一 GCC/Clang 与 MSVC 的架构宏差异；所有源文件一律使用
// #if RSE_ARCH_X86_64 / #if RSE_ARCH_AARCH64 判断。
#if defined(__x86_64__) || defined(_M_X64)
#define RSE_ARCH_X86_64 1
#else
#define RSE_ARCH_X86_64 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define RSE_ARCH_AARCH64 1
#else
#define RSE_ARCH_AARCH64 0
#endif

// --- 符号导出控制 ---
// 库的非内联导出符号（gf8 的批量运算入口）统一标注 RSE_API：
//   - ELF/Mach-O：配合 -fvisibility=hidden（见 CMakeLists.txt），
//     只有标注的符号进入动态符号表，缩小二进制、加快加载、
//     避免符号冲突；
//   - Windows：编译动态库时 dllexport，使用动态库时 dllimport
//     （消费者通过 CMake 导出目标自动获得 RSE_SHARED 定义）。
#if defined(_WIN32) && defined(RSE_SHARED)
#if defined(RSE_BUILDING)
#define RSE_API __declspec(dllexport)
#else
#define RSE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define RSE_API __attribute__((visibility("default")))
#else
#define RSE_API
#endif
