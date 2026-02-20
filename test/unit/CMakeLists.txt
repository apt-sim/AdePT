# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25.2)

project(testAMD
  VERSION 0.1.0
  DESCRIPTION "Accelerated demonstrator of electromagnetic Particle Transport"
  LANGUAGES C CXX HIP)

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

find_package(hip REQUIRED)

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

#----------------------------------------------------------------------------#
# Helper Macros/Functions
#----------------------------------------------------------------------------#
macro(build_tests TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST})
    #set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    #cuda_rdc_target_link_libraries(${TARGET_NAME} AdePT_G4_integration)
    target_include_directories(${TARGET_NAME} PRIVATE /home/gjuan/Dev/AdePT_dev/include/)
    target_link_libraries(${TARGET_NAME} PRIVATE hip::host hip::device)
   endforeach()
endmacro()

macro(add_to_test TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
   endforeach()
endmacro()

add_library(CopCore INTERFACE)
target_include_directories(CopCore
  INTERFACE
    $<BUILD_INTERFACE:/home/gjuan/Dev/AdePT_dev/include/AdePT/copcore/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)

#----------------------------------------------------------------------------#
# Common Data
#----------------------------------------------------------------------------#
set(TESTING_GDML "${PROJECT_BINARY_DIR}/trackML.gdml")
file(DOWNLOAD https://gitlab.cern.ch/VecGeom/VecGeom/raw/v1.2.0/persistency/gdml/gdmls/trackML.gdml
  "${TESTING_GDML}"
  EXPECTED_HASH SHA256=2c53e0f2f4673c61f8679702532647bf71ec64c1613aae330fa835e7d7087064
)

set(CMS2018_GDML "${PROJECT_BINARY_DIR}/cms2018.gdml")
file(DOWNLOAD https://gitlab.cern.ch/VecGeom/VecGeom/raw/v1.2.0/persistency/gdml/gdmls/cms2018.gdml
  "${CMS2018_GDML}"
  EXPECTED_HASH SHA256=a6538d8f196fbfe4c14e806df3439d5a0d7050d538d364faabe882c750970e63
)

#----------------------------------------------------------------------------#
# Detailed tests
#----------------------------------------------------------------------------#
# add_subdirectory(testField)

#----------------------------------------------------------------------------#
# Link/RDC tests
#----------------------------------------------------------------------------#
# - Check CopCore links as expected
#add_executable(test_copcore_link test_copcore_link.cpp)
#target_include_directories(test_copcore_link PRIVATE ${PROJECT_SOURCE_DIR}/include)
#target_include_directories(test_copcore_link PRIVATE /home/gjuan/Dev/AdePT_dev/include/)
#target_link_libraries(test_copcore_link PRIVATE CopCore)

#----------------------------------------------------------------------------#
# Basic Unit Tests
#----------------------------------------------------------------------------#
set(ADEPT_UNIT_TESTS_BASE
  test_atomic.cpp               # Unit test for atomic ops
  #test_ranluxpp.hip             # Unit test for RANLUX++
  test_queue.cpp                # Unit test for mpmc_bounded_queue
  #test_track_block.hip          # Unit test for BlockData
  #test_magfieldRK.cpp          # Unit test for Mag-Field integration classes
  #compare_integration_precision.cpp # Script for generating float vs double trajectories
  #test_hostTrackDataMapper.cpp # Unit test for hostTrackDataMapper
  # test_Bfield_step_debugger.cpp # experimental test to debug single stuck rays in AdePT
)

#if(ADEPT_USE_HIP)
#  set_source_files_properties(test_atomic.cu PROPERTIES LANGUAGE HIP)
#  set_source_files_properties(test_ranluxpp.cu PROPERTIES LANGUAGE HIP)
#  set_source_files_properties(test_track_block.cu PROPERTIES LANGUAGE HIP)
#  set_source_files_properties(test_queue.cu PROPERTIES LANGUAGE HIP)
#endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;>")
add_compile_options("$<$<CONFIG:Debug>:-g;-fstandalone-debug;>")

build_tests("${ADEPT_UNIT_TESTS_BASE}")
add_to_test("${ADEPT_UNIT_TESTS_BASE}")
