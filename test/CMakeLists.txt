# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Helper Macros/Functions
macro(build_tests TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
      $<INSTALL_INTERFACE:base>
    )
    target_link_libraries(${TARGET_NAME} VecCore::VecCore CopCore::CopCore)
    target_compile_options(${TARGET_NAME} PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
   endforeach()
endmacro()

macro(add_to_test TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
   endforeach()
endmacro()

# - Minimal test of cuda compilation
add_executable(example_add example_add.cu)

# - Test that linkage to CopCore target works
add_executable(test_copcore_link test_copcore_link.cpp)
target_link_libraries(test_copcore_link PRIVATE VecCore::VecCore CopCore::CopCore)

# - Unit tests
set(ADEPT_UNIT_TESTS_BASE
  test_atomic.cu               # Unit test for atomic ops
  test_queue.cu                # Unit test for mpmc_bounded_queue
  test_track_block.cu          # Unit test for BlockData
)

build_tests("${ADEPT_UNIT_TESTS_BASE}")
add_to_test("${ADEPT_UNIT_TESTS_BASE}")

# - Examples
# Noddy example of particle processing on CPU
add_executable(fisher_price fisher_price.cpp)
# Noddy example of particle processing with GPU
add_executable(cufisher_price cufisher_price.cu)
target_include_directories(cufisher_price PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(cufisher_price PRIVATE CUDA::curand VecCore::VecCore CopCore::CopCore)
