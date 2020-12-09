# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Example 1 of particle processing with GPU
add_executable(example1 example1.cu)
target_include_directories(example1 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/physics/processes/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tracking/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(example1 PRIVATE CUDA::curand VecCore::VecCore VecGeom::vecgeom CopCore::CopCore)
