# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Example 2 of particle processing with GPU, including geometry
# and reproducible results using one RANLUX++ state per track.
add_executable(example2 example2.cpp example2.cu)
target_include_directories(example2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/physics/processes/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tracking/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(example2 PRIVATE CUDA::curand VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml CopCore::CopCore)
set_target_properties(example2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
