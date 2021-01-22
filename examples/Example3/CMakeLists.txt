# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)

project(Example3 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Disable exceptions and RTTI for better performance and
# to ensure we don't use these features during development
add_compile_options(-fno-exceptions -fno-rtti)

if(BUILD_TESTING)
  enable_testing()
endif()

add_executable(example3
  common.cc
  field.cc
  geometry.cc
  init.cc
  loop.cc
  main.cc
  particle.cc
  physics.cc
  ppm.cc
  primitives.cc
  random.cc
  scoring.cc
  transport.cc
  user.cc
  verlet.cc)

add_test(NAME example3 COMMAND example3 1024)

add_executable(example3-cuda
  common.cu
  field.cu
  geometry.cu
  init.cu
  loop.cu
  main.cu
  particle.cu
  physics.cu
  ppm.cu
  primitives.cu
  random.cu
  scoring.cu
  transport.cu
  user.cu
  verlet.cu)

target_link_libraries(example3-cuda CUDA::cudart)
set_target_properties(example3-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME example3-cuda COMMAND example3-cuda 1024)

add_executable(example3-raymarch
  ppm.cc
  primitives.cc
  render.cc
  raymarch.cc)

add_test(NAME example3-raymarch COMMAND example3-raymarch 1024 768 raymarch.ppm)

add_executable(example3-raymarch-cuda
  ppm.cu
  primitives.cu
  render.cu
  raymarch.cu)

target_link_libraries(example3-raymarch-cuda CUDA::cudart)
set_target_properties(example3-raymarch-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME example3-raymarch-cuda COMMAND example3-raymarch 1024 768 raymarch-cuda.ppm)
