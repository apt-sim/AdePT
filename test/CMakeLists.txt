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
    target_compile_options(${TARGET_NAME} PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;--extended-lambda>")
   endforeach()
endmacro()

macro(add_to_test TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
   endforeach()
endmacro()

# - Test that linkage to CopCore target works
add_executable(test_copcore_link test_copcore_link.cpp)
target_link_libraries(test_copcore_link PRIVATE VecCore::VecCore CopCore::CopCore)

# - Unit tests
set(ADEPT_UNIT_TESTS_BASE
  test_atomic.cu               # Unit test for atomic ops
  test_queue.cu                # Unit test for mpmc_bounded_queue
  test_track_block.cu          # Unit test for BlockData
)

add_executable(test_launcher test_launcher.cu test_launcher.cpp)
target_include_directories(test_launcher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(test_launcher VecCore::VecCore CopCore::CopCore)
target_compile_options(test_launcher PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;--extended-lambda>")

build_tests("${ADEPT_UNIT_TESTS_BASE}")
add_to_test("${ADEPT_UNIT_TESTS_BASE}")
