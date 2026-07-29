cmake_minimum_required(VERSION 3.20)
project(amd_rocm_dma_gesture_accel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(dma_engine src/dma_kernel.cpp src/gesture_cnn.cpp)
add_executable(test_dma tests/test_dma_transfer.cpp)
target_link_libraries(test_dma PRIVATE dma_engine)
