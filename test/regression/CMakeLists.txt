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
file(COPY "${PROJECT_SOURCE_DIR}/examples/data/testEm3.gdml" DESTINATION "${PROJECT_BINARY_DIR}")
set(TESTING_GDML "${PROJECT_BINARY_DIR}/testEm3.gdml")

#----------------------------------------------------------------------------#
# G4 App-based Tests
#----------------------------------------------------------------------------#

add_subdirectory(IntegrationTest)

# Scripts
set(SCRIPTS_DIR "${PROJECT_SOURCE_DIR}/test/regression/scripts")
set(TEMP_DIR "${PROJECT_SOURCE_DIR}/test/regression/ci_tmp")

# This test checks the reproducibility of AdePT by running 8 ttbar events and checking that the energy deposition is exactly the same.
add_test(NAME reproducibility_cms_ttbar
    COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/reproducibility.sh
    "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

# test that compares the physics output of a full AdePT run against a high-statistics Geant4 simulation using G4HepEm.
# The energy deposition per layer error must be < 1% to pass the test
add_test(NAME testEm3_validation
    COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/validation_testem3.sh
    "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

set(APP_TESTS

)