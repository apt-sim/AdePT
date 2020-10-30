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

# - Test that linkage to CopCore target works
add_executable(test_copcore_link test_copcore_link.cpp)
target_link_libraries(test_copcore_link PRIVATE VecCore::VecCore CopCore::CopCore)

# - Unit tests
set(ADEPT_UNIT_TESTS_BASE
  test_atomic.cu               # Unit test for atomic ops
  test_queue.cu                # Unit test for mpmc_bounded_queue
  test_track_block.cu          # Unit test for BlockData
  #graph_executor.cu
  #kernel_executor.cu
  #test_executor.cu
)

set(ADEPT_EXECUTOR_SRCS
  test_executor.cu
  test_executor.cpp
)

add_executable(test_executor ${ADEPT_EXECUTOR_SRCS})
target_include_directories(test_executor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(test_executor VecCore::VecCore CopCore::CopCore)
target_compile_options(test_executor PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

build_tests("${ADEPT_UNIT_TESTS_BASE}")
add_to_test("${ADEPT_UNIT_TESTS_BASE}")

# - Specific executable options
#target_compile_options(graph_executor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -gencode arch=compute_60,code=sm_60>)
#target_compile_options(kernel_executor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -gencode arch=compute_60,code=sm_60>)
#target_compile_options(smart_executor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -gencode arch=compute_60,code=sm_60>)
target_compile_options(test_executor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda 
 -gencode=arch=compute_50,code=sm_50
 -gencode=arch=compute_52,code=sm_52
 -gencode=arch=compute_60,code=sm_60
 -gencode=arch=compute_61,code=sm_61
 -gencode=arch=compute_70,code=sm_70
 -gencode=arch=compute_75,code=sm_75
 -gencode=arch=compute_75,code=compute_75>
 )
#target_link_libraries(smart_executor VecGeom::vecgeom)
