# bench/CMakeLists.txt - 基准测试目标的构建配置。
# rse_bandwidth 是编码带宽基准（对应 Rust 版的 benches/bandwidth.rs），
# 链接静态库以避免动态调用开销影响测量。
add_executable(rse_bandwidth bandwidth.cpp)
target_link_libraries(rse_bandwidth PRIVATE reed_solomon_erasure_static)
