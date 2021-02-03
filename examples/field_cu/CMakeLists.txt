# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

add_executable(fieldStep field_step.cu)
target_include_directories(fieldStep PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tracking/inc>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/magneticfield/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(fieldStep VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml CopCore::CopCore)
target_compile_options(fieldStep PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(fieldStep PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

