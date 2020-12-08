# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Noddy example 1 of particle processing with GPU
add_executable(cufisher_price cufisher_price.cu)
target_include_directories(cufisher_price PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(cufisher_price PRIVATE CUDA::curand VecCore::VecCore CopCore::CopCore_CUDA)

# Noddy example 2 of particle processing with GPU
add_executable(cufisher_price_v2 cufisher_price_v2.cu)
target_include_directories(cufisher_price_v2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
  )
target_link_libraries(cufisher_price_v2 PRIVATE CUDA::curand VecCore::VecCore CopCore::CopCore_CUDA)

