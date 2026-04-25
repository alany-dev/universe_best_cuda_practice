cmake_minimum_required(VERSION 3.20.0)
project(cuda_best_practice VERSION 0.1.0 LANGUAGES CUDA CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit)

option(ENABLE_PERF_TEST "Enable performance test" ON)

# 如果开启，自动给所有 CUDA 文件定义宏 PERF_TEST_ENABLED
if(ENABLE_PERF_TEST)
    add_compile_definitions(PERF_TEST_ENABLED)
    message(STATUS "Performance test: ENABLED")
else()
    message(STATUS "Performance test: DISABLED")
endif()


add_subdirectory(1_cuda_reduce_study)