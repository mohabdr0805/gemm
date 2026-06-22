cmake_minimum_required(VERSION 3.18)
project(gemm_optim LANGUAGES CXX)

option(USE_CUDA "Build the CUDA implementation (requires an NVIDIA GPU + nvcc)" OFF)
option(GEMM_NATIVE "Tune the build for the host CPU (-march=native); turn OFF for portable binaries/images" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  if(GEMM_NATIVE)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native")
  endif()
endif()

find_package(OpenMP)

# --- CPU library -------------------------------------------------------------
add_library(gemm_cpu src/gemm_cpu.cpp)
target_include_directories(gemm_cpu PUBLIC include)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gemm_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- CUDA library (optional) -------------------------------------------------
if(USE_CUDA)
  enable_language(CUDA)
  add_library(gemm_cuda src/gemm_cuda.cu)
  target_include_directories(gemm_cuda PUBLIC include)
  # 75 = Turing (GTX 1650), 86 = Ampere (RTX 30xx)
  set_target_properties(gemm_cuda PROPERTIES CUDA_ARCHITECTURES "75;86")
endif()

# --- Benchmark ---------------------------------------------------------------
add_executable(bench benchmarks/bench.cpp)
target_link_libraries(bench PRIVATE gemm_cpu)
if(USE_CUDA)
  target_link_libraries(bench PRIVATE gemm_cuda)
  target_compile_definitions(bench PRIVATE USE_CUDA)
endif()

# --- Tests -------------------------------------------------------------------
enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE gemm_cpu)
if(USE_CUDA)
  target_link_libraries(test_correctness PRIVATE gemm_cuda)
  target_compile_definitions(test_correctness PRIVATE USE_CUDA)
endif()
add_test(NAME correctness COMMAND test_correctness)
