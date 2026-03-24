#
# CMakeLists for CUDA Debug Test
# This builds a standalone test program for cuda-gdb debugging
#

cmake_minimum_required(VERSION 3.20)

project(CudaRasterizerTest LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

# Enable debug symbols for cuda-gdb
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Find CUDA
find_package(CUDA REQUIRED)

# Find Python (required for PyTorch)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# Explicitly find Python library
find_library(PYTHON_LIBRARY 
    NAMES python3.10 python310
    HINTS ${Python3_ROOT_DIR} ${CONDA_PREFIX} /usr/lib
)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Build the CudaRasterizer library first
add_library(CudaRasterizer
    cuda_rasterizer/backward.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/rasterizer.h
)

set_target_properties(CudaRasterizer PROPERTIES 
    CUDA_ARCHITECTURES "70;75;86"
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(CudaRasterizer PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Build the test executable
add_executable(test_cuda_debug 
    test_cuda_debug.cpp
)

target_link_libraries(test_cuda_debug 
    CudaRasterizer
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${TORCH_LIBRARIES}
    ${PYTHON_LIBRARY}
)

# Set C++ standard for PyTorch
set_property(TARGET test_cuda_debug PROPERTY CXX_STANDARD 17)

target_include_directories(test_cuda_debug PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${Python3_INCLUDE_DIRS}
)

# Set CUDA flags for the test executable
set_target_properties(test_cuda_debug PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
