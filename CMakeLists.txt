# reed_solomon_erasure - Reed-Solomon 纠删码库（C++23）
#
# 同一份源码同时产出静态库与动态库（共享一个 PIC 对象库，只编译一次），
# 并安装 CMake 包配置，下游可通过 find_package(reed_solomon_erasure)
# 使用 rse::shared / rse::static 目标。
cmake_minimum_required(VERSION 3.23)

project(reed_solomon_erasure
    VERSION 6.0.0
    DESCRIPTION "C++23 implementation of Reed-Solomon erasure coding"
    LANGUAGES CXX)

option(RSE_BUILD_TESTS "Build unit tests" ON)
option(RSE_BUILD_BENCH "Build benchmarks" OFF)

# 未指定构建类型时默认 Release（SIMD 内核的性能依赖优化等级）。
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 目标架构检测（决定编译哪些 SIMD 内核翻译单元）---

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _rse_arch)
set(RSE_ARCH_X86_64 FALSE)
set(RSE_ARCH_AARCH64 FALSE)
if(_rse_arch MATCHES "^(x86_64|amd64|x64)$")
    set(RSE_ARCH_X86_64 TRUE)
elseif(_rse_arch MATCHES "^(aarch64|arm64)$")
    set(RSE_ARCH_AARCH64 TRUE)
endif()

# --- 库本体 ---

set(RSE_SOURCES
    src/galois_8.cpp
)

if(RSE_ARCH_X86_64)
    # x86-64：三个 SIMD 内核全部编译进库，运行时按 CPUID 选择。
    # 每个内核单独成翻译单元，仅给该文件开启对应指令集的编译选项，
    # 这样库的其余部分不依赖任何 -march，老 CPU 上也能安全运行。
    list(APPEND RSE_SOURCES
        src/gf8_simd_ssse3.cpp
        src/gf8_simd_avx2.cpp
        src/gf8_simd_avx512.cpp
    )
    if(MSVC)
        # MSVC 没有独立的 SSSE3 开关，x64 下 intrinsics 默认可用。
        set_source_files_properties(src/gf8_simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties(src/gf8_simd_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    else()
        set_source_files_properties(src/gf8_simd_ssse3.cpp PROPERTIES COMPILE_OPTIONS "-mssse3")
        set_source_files_properties(src/gf8_simd_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
        # _mm512_shuffle_epi8 需要 AVX-512BW，故 F 与 BW 一起开。
        set_source_files_properties(src/gf8_simd_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw")
    endif()
elseif(RSE_ARCH_AARCH64)
    # AArch64：NEON 是基线特性，无需指令集开关，也无需运行时检测。
    list(APPEND RSE_SOURCES src/gf8_simd_neon.cpp)
    if(NOT MSVC)
        # 部分工具链（如 OpenBSD clang）在 AArch64 上默认 +strict-align，
        # 会把内核中的非对齐向量加载标量化成逐字节加载，性能掉一个数量级。
        # 用户态访问普通内存允许非对齐，所以对这个翻译单元显式关闭
        # 严格对齐。（在默认就不严格对齐的 Linux 工具链上等价于无操作。）
        set_source_files_properties(src/gf8_simd_neon.cpp
            PROPERTIES COMPILE_OPTIONS "-mno-strict-align")
    endif()
endif()

# galois_8.hpp 中用 consteval 在编译期生成 64KiB 的 GF(2^8) 乘法表，
# 单次常量求值的运算量超过 clang 与 MSVC 的默认步数上限（GCC 的默认
# 上限足够）。这里提高上限，并通过导出目标的 INTERFACE 选项传播给
# 所有库的使用者。
set(RSE_CONSTEXPR_FLAGS
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fconstexpr-steps=8000000>
    $<$<CXX_COMPILER_ID:MSVC>:/constexpr:steps8000000>
)

# 源码只编译一次（PIC 对象库），静态库与动态库共享同一批对象文件。
add_library(rse_objects OBJECT ${RSE_SOURCES})
set_target_properties(rse_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(rse_objects PRIVATE ${RSE_CONSTEXPR_FLAGS})
target_include_directories(rse_objects
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(rse_objects PUBLIC cxx_std_23)

add_library(reed_solomon_erasure SHARED $<TARGET_OBJECTS:rse_objects>)
add_library(reed_solomon_erasure_static STATIC $<TARGET_OBJECTS:rse_objects>)

foreach(tgt reed_solomon_erasure reed_solomon_erasure_static)
    target_include_directories(${tgt}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_compile_features(${tgt} PUBLIC cxx_std_23)
    # 下游编译我们的头文件同样需要更高的 constexpr 步数上限。
    target_compile_options(${tgt} INTERFACE ${RSE_CONSTEXPR_FLAGS})
    set_target_properties(${tgt} PROPERTIES OUTPUT_NAME reed_solomon_erasure)
endforeach()

# 动态库版本：libreed_solomon_erasure.so.6.0.0，SONAME 带主版本号。
set_target_properties(reed_solomon_erasure PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_library(rse::shared ALIAS reed_solomon_erasure)
add_library(rse::static ALIAS reed_solomon_erasure_static)

# --- 测试 ---

if(RSE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- 基准测试 ---

if(RSE_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# --- 安装与包导出 ---

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS reed_solomon_erasure reed_solomon_erasure_static
    EXPORT reed_solomon_erasure-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/reed_solomon_erasure
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT reed_solomon_erasure-targets
    NAMESPACE rse::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/reed_solomon_erasure
)

configure_package_config_file(
    cmake/reed_solomon_erasure-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/reed_solomon_erasure-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/reed_solomon_erasure
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/reed_solomon_erasure-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/reed_solomon_erasure-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/reed_solomon_erasure-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/reed_solomon_erasure
)
