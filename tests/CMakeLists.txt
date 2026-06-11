# tests/CMakeLists.txt - 测试目标的构建配置。
# rse_tests 是基于 doctest 的主测试程序，链接静态库；
# rse_shared_link_check 用于验证共享库可正常链接运行。
add_executable(rse_tests
    test_main.cpp
    test_errors.cpp
    test_galois_8.cpp
    test_galois_16.cpp
    test_matrix.cpp
    test_reed_solomon.cpp
    test_reed_solomon_gf16.cpp
)

target_link_libraries(rse_tests PRIVATE reed_solomon_erasure_static)
target_include_directories(rse_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME rse_tests COMMAND rse_tests)

# 同时确认共享库能够正确地链接进可执行文件。
add_executable(rse_shared_link_check shared_link_check.cpp)
target_link_libraries(rse_shared_link_check PRIVATE reed_solomon_erasure)
add_test(NAME rse_shared_link_check COMMAND rse_shared_link_check)
