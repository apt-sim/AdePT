# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Example 4 of particle processing with GPU, including geometry, magnetic field
# and reproducible results using one RANLUX++ state per track.
add_executable(example4 example4.cpp example4.cu)
target_include_directories(example4 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/physics/processes/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tracking/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/magneticfield/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(example4 PRIVATE VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml CopCore::CopCore)
target_compile_options(example4 PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(example4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
