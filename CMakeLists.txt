cmake_minimum_required(VERSION 3.20) # 3.20: `ctest --test-dir` (used in docs/CI)
project(gemm_optim LANGUAGES CXX)

option(USE_CUDA "Build the CUDA implementation (requires an NVIDIA GPU + nvcc)" OFF)
option(GEMM_NATIVE "Tune the build for the host CPU (-march=native / MSVC /arch:AVX2); turn OFF for portable binaries/images" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Keep /DNDEBUG resp. -DNDEBUG: overwriting the default Release flags without it
# would silently re-enable assert() in release builds.
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  # MSVC's x64 default is SSE2 (4-wide, no FMA): without /arch:AVX2 the tiled
  # GEMM's vectorizable j-loop runs at half width and unfused mul+add. Guarded
  # by GEMM_NATIVE for the same reason -march=native is: AVX2 binaries crash on
  # pre-2013 CPUs, so portable/Docker builds keep the SSE2 baseline.
  if(GEMM_NATIVE)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /arch:AVX2")
  endif()
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  if(GEMM_NATIVE)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native")
  endif()
endif()

find_package(OpenMP)

# --- CPU library -------------------------------------------------------------
add_library(gemm_cpu src/gemm_cpu.cpp src/softmax_cpu.cpp src/attention_cpu.cpp)
target_include_directories(gemm_cpu PUBLIC include)
# MSVC's classic /openmp is stuck at OpenMP 2.0, which silently ignores the
# collapse(2) in gemm_tiled (warning C4849) and parallelizes only the outer loop.
# /openmp:llvm selects MSVC's OpenMP 3.1+ runtime so the (ii,jj) tiles are really
# spread across threads. GCC/Clang keep the normally-detected OpenMP.
if(MSVC)
  # PRIVATE: the OpenMP pragmas live only in the .cpp files (no public header
  # uses OpenMP), and MSVC objects embed the libomp defaultlib directive, so
  # consumers link fine without inheriting the flag.
  # /openmp:experimental = the LLVM OpenMP runtime (so collapse(2) really
  # parallelizes, like /openmp:llvm) PLUS the `omp simd` directive, which the
  # tiled GEMM needs to vectorize its inner loop (MSVC refuses on its own,
  # assuming C may alias B).
  target_compile_options(gemm_cpu PRIVATE /openmp:experimental)
elseif(OpenMP_CXX_FOUND)
  # PUBLIC on purpose here: with GCC/Clang the consumers of the static lib do
  # need -fopenmp at link time to resolve the GOMP/omp symbols.
  target_link_libraries(gemm_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- CUDA library (optional) -------------------------------------------------
if(USE_CUDA)
  # Default archs: 75 = Turing (GTX 1650), 86 = Ampere (RTX 30xx). Only a
  # default: -DCMAKE_CUDA_ARCHITECTURES=89 (Ada, etc.) on the command line wins.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED) # for CUDA::cublas (the benchmark baseline)
  add_library(gemm_cuda src/gemm_cuda.cu src/softmax_cuda.cu src/attention_cuda.cu)
  target_include_directories(gemm_cuda PUBLIC include)
  # PRIVATE is enough: CMake still adds cublas to the final link of consumers
  # of this static library; no cuBLAS type appears in our public headers.
  target_link_libraries(gemm_cuda PRIVATE CUDA::cublas)
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

add_executable(test_softmax tests/test_softmax.cpp)
target_link_libraries(test_softmax PRIVATE gemm_cpu)
if(USE_CUDA)
  target_link_libraries(test_softmax PRIVATE gemm_cuda)
  target_compile_definitions(test_softmax PRIVATE USE_CUDA)
endif()
add_test(NAME softmax COMMAND test_softmax)

add_executable(test_attention tests/test_attention.cpp)
target_link_libraries(test_attention PRIVATE gemm_cpu)
if(USE_CUDA)
  target_link_libraries(test_attention PRIVATE gemm_cuda)
  target_compile_definitions(test_attention PRIVATE USE_CUDA)
endif()
add_test(NAME attention COMMAND test_attention)
