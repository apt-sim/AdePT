# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

#----------------------------------------------------------------------------#
# Helper Macros/Functions
#----------------------------------------------------------------------------#
macro(build_tests TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    add_executable(${TARGET_NAME} ${TEST})
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${TARGET_NAME} AdePT_G4_integration)
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
add_subdirectory(testField)

#----------------------------------------------------------------------------#
# Link/RDC tests
#----------------------------------------------------------------------------#
# - Check CopCore links as expected 
add_executable(test_copcore_link test_copcore_link.cpp)
target_include_directories(test_copcore_link PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_copcore_link PRIVATE CopCore)

# - Check G4VG links as expected 
if(VecGeom_SURF_FOUND)
  message(STATUS "${Magenta}Disabled g4vg when using VecGeom surface model${ColorReset}")
else()
  message(STATUS "Fetching and compiling g4vg ...")
  include(FetchContent)
  FetchContent_Declare(
    g4vg
    EXCLUDE_FROM_ALL
    URL https://github.com/celeritas-project/g4vg/archive/e034ada099132f857866949ec9ce8eadf1ff2251.zip)
  FetchContent_MakeAvailable(g4vg)

  add_executable(test_g4vg_link test_g4vg_link.cpp)
  target_link_libraries(test_g4vg_link PRIVATE G4VG::g4vg)
endif()

#----------------------------------------------------------------------------#
# Basic Unit Tests
#----------------------------------------------------------------------------#
set(ADEPT_UNIT_TESTS_BASE
  test_atomic.cu               # Unit test for atomic ops
  test_ranluxpp.cu             # Unit test for RANLUX++
  test_queue.cu                # Unit test for mpmc_bounded_queue
  test_track_block.cu          # Unit test for BlockData
  test_magfieldRK.cpp          # Unit test for Mag-Field integration classes
)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;>")

build_tests("${ADEPT_UNIT_TESTS_BASE}")
add_to_test("${ADEPT_UNIT_TESTS_BASE}")
