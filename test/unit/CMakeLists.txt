# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

#----------------------------------------------------------------------------#
# Helper Macros/Functions
#----------------------------------------------------------------------------#
function(build_header_only_tests)
  foreach(TEST IN LISTS ARGN)
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST})
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  endforeach()
endfunction()

function(build_linked_tests LINK_TARGET)
  foreach(TEST IN LISTS ARGN)
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST})
    #set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    cuda_rdc_target_link_libraries(${TARGET_NAME} PRIVATE ${LINK_TARGET})
  endforeach()
endfunction()

include(GoogleTest)

function(build_gtest_tests LINK_TARGET)
  foreach(TEST IN LISTS ARGN)
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST})
    cuda_rdc_target_link_libraries(${TARGET_NAME} PRIVATE ${LINK_TARGET} GTest::gtest_main)
    gtest_discover_tests(${TARGET_NAME}
      DISCOVERY_MODE PRE_TEST
      PROPERTIES LABELS "unit")
  endforeach()
endfunction()

function(add_to_test)
  foreach(TEST IN LISTS ARGN)
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
  endforeach()
endfunction()

#----------------------------------------------------------------------------#
# Common Data
#----------------------------------------------------------------------------#
# Keep unit tests aligned with regression tests by using the in-repo geometry.
configure_file(${PROJECT_SOURCE_DIR}/examples/data/cms2018_sd.gdml ${PROJECT_BINARY_DIR}/cms2018_sd.gdml)

#----------------------------------------------------------------------------#
# Basic Unit Tests
#----------------------------------------------------------------------------#
set(ADEPT_HEADER_ONLY_UNIT_TESTS
  test_header_only_include.cpp  # Include smoke test for header-only AdePT utilities
  test_atomic.cu               # Unit test for atomic ops
  test_ranluxpp.cu             # Unit test for RANLUX++
  test_queue.cu                # Unit test for mpmc_bounded_queue
  test_track_block.cu          # Unit test for BlockData
)

set(ADEPT_TRANSPORT_UNIT_TESTS
  test_hostCircularBuffer.cpp  # Unit test for HostCircularBuffer
  # test_Bfield_step_debugger.cpp # experimental test to debug single stuck rays in AdePT
)

set(ADEPT_TRANSPORT_GTESTS
  test_bfield.cpp              # Deterministic B-field integration tests
  test_bfield_navigation.cpp   # B-field RK/navigation handoff tests
)

set(ADEPT_TRANSPORT_UTILITIES
  compare_integration_precision.cpp # Diagnostic utility for generating float vs double trajectories
)

set(ADEPT_G4_INTEGRATION_UNIT_TESTS
  test_hostTrackDataMapper.cpp # Unit test for hostTrackDataMapper
)

if(ADEPT_USE_EXT_BFIELD)
  list(APPEND ADEPT_TRANSPORT_UNIT_TESTS
    test_general_magnetic_field.cu # Unit test for device-side Covfie field evaluation
  )
  list(APPEND ADEPT_G4_INTEGRATION_UNIT_TESTS
    test_covfie_field.cpp       # Unit test for Geant4-side Covfie field loading
  )
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;>")

build_header_only_tests(${ADEPT_HEADER_ONLY_UNIT_TESTS})
build_linked_tests(AdePT::Transport ${ADEPT_TRANSPORT_UNIT_TESTS} ${ADEPT_TRANSPORT_UTILITIES})
build_gtest_tests(AdePT::Transport ${ADEPT_TRANSPORT_GTESTS})
build_linked_tests(AdePT::G4Integration ${ADEPT_G4_INTEGRATION_UNIT_TESTS})

if(ADEPT_USE_EXT_BFIELD)
  foreach(TARGET_NAME test_covfie_field test_general_magnetic_field)
    target_include_directories(${TARGET_NAME}
      PRIVATE
        ${PROJECT_SOURCE_DIR}/test/common
        ${PROJECT_SOURCE_DIR}/examples/common/include
    )
  endforeach()
endif()

set(ADEPT_UNIT_TESTS
  ${ADEPT_HEADER_ONLY_UNIT_TESTS}
  ${ADEPT_TRANSPORT_UNIT_TESTS}
  ${ADEPT_G4_INTEGRATION_UNIT_TESTS}
)

add_to_test(${ADEPT_UNIT_TESTS})

set(ADEPT_UNIT_TEST_NAMES
  test_header_only_include
  test_atomic
  test_ranluxpp
  test_queue
  test_track_block
  test_hostCircularBuffer
  test_hostTrackDataMapper
)
if(ADEPT_USE_EXT_BFIELD)
  list(APPEND ADEPT_UNIT_TEST_NAMES
    test_covfie_field
    test_general_magnetic_field
  )
endif()
set_tests_properties(${ADEPT_UNIT_TEST_NAMES} PROPERTIES LABELS "unit")
