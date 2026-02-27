# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

#----------------------------------------------------------------------------#
# Helper Macros/Functions
#----------------------------------------------------------------------------#
macro(build_tests TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST})
    #set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    cuda_rdc_target_link_libraries(${TARGET_NAME} AdePT_G4_integration)
   endforeach()
endmacro()

macro(add_to_test TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
   endforeach()
endmacro()

#----------------------------------------------------------------------------#
# Common Data
#----------------------------------------------------------------------------#
# Keep unit tests aligned with regression tests by using the in-repo geometry.
configure_file(${PROJECT_SOURCE_DIR}/examples/data/cms2018_sd.gdml ${PROJECT_BINARY_DIR}/cms2018_sd.gdml)

#----------------------------------------------------------------------------#
# Detailed tests
#----------------------------------------------------------------------------#
add_subdirectory(testField)

#----------------------------------------------------------------------------#
# Link/RDC tests
#----------------------------------------------------------------------------#
# - Check CopCore links as expected
add_executable(test_copcore_link test_copcore_link.cpp)
target_include_directories(test_copcore_link PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_copcore_link PRIVATE CopCore)

#----------------------------------------------------------------------------#
# Basic Unit Tests
#----------------------------------------------------------------------------#
set(ADEPT_UNIT_TESTS_BASE
  test_atomic.cu               # Unit test for atomic ops
  test_ranluxpp.cu             # Unit test for RANLUX++
  test_queue.cu                # Unit test for mpmc_bounded_queue
  test_track_block.cu          # Unit test for BlockData
  test_magfieldRK.cpp          # Unit test for Mag-Field integration classes
  compare_integration_precision.cpp # Script for generating float vs double trajectories
  test_hostTrackDataMapper.cpp # Unit test for hostTrackDataMapper
  # test_Bfield_step_debugger.cpp # experimental test to debug single stuck rays in AdePT
)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;>")

build_tests("${ADEPT_UNIT_TESTS_BASE}")
add_to_test("${ADEPT_UNIT_TESTS_BASE}")
