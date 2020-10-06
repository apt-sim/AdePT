# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# - Test that linkage to CopCore target works
add_executable(test_copcore_link test_copcore_link.cpp)
target_link_libraries(test_copcore_link PRIVATE CopCore::CopCore)


# - Minimal test of cuda compilation
add_executable(example_add example_add.cu)

# Noddy example of particle processing on CPU
add_executable(fisher_price fisher_price.cpp)
# Noddy example of particle processing with GPU
add_executable(cufisher_price cufisher_price.cu)
target_include_directories(cufisher_price PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
  )
target_link_libraries(cufisher_price PRIVATE CUDA::curand VecCore::VecCore VecGeom::vecgeom)
# Unit test for atomic ops
add_executable(test_atomic test_atomic.cu)
target_include_directories(test_atomic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(test_atomic VecCore::VecCore VecGeom::vecgeom)
add_test(NAME test_atomic COMMAND test_atomic)

# Unit test for BlockData
add_executable(test_track_block test_track_block.cu)
target_include_directories(test_track_block PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(test_track_block VecCore::VecCore VecGeom::vecgeom)
add_test(NAME test_track_block COMMAND test_track_block)

# Unit test for mpmc_bounded_queue
add_executable(test_queue test_queue.cu)
target_include_directories(test_queue PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(test_queue VecCore::VecCore VecGeom::vecgeom)
add_test(NAME test_queue COMMAND test_queue)
