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
# G4 App-based Tests
#----------------------------------------------------------------------------#

# Move files needed for testing to the build directory. These files are stored under examples/data, but eventually may be moved to a common directory
configure_file(${PROJECT_SOURCE_DIR}/examples/data/cms2018_sd.gdml ${PROJECT_BINARY_DIR}/cms2018_sd.gdml)
configure_file(${PROJECT_SOURCE_DIR}/examples/data/ppttbar.hepmc3 ${PROJECT_BINARY_DIR}/ppttbar.hepmc3)

add_subdirectory(IntegrationTest)

# Scripts
set(SCRIPTS_DIR "${PROJECT_SOURCE_DIR}/test/regression/scripts")
set(TEMP_DIR "${PROJECT_SOURCE_DIR}/test/regression/ci_tmp")
set(TEMP_SPLIT_DIR "${PROJECT_SOURCE_DIR}/test/regression/ci_tmp_split")

# This test checks the reproducibility of AdePT by running 8 ttbar events and checking that the energy deposition is exactly the same.
add_test(NAME reproducibility_cms_ttbar
    COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/reproducibility.sh
    "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
set_tests_properties(reproducibility_cms_ttbar PROPERTIES LABELS "validation")

# This test checks the reproducibility of AdePT with transport only in some regions by running 50 e- events in the testEm3 geometry
# and checking that the energy deposition is exactly the same.
add_test(NAME reproducibility_regions
    COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/reproducibility_regions.sh
    "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
set_tests_properties(reproducibility_regions PROPERTIES LABELS "validation")

# Same test as above but using Woodcock tracking in the testEm3 geometry
add_test(NAME reproducibility_WDT
    COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/reproducibility_WDT.sh
    "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
set_tests_properties(reproducibility_WDT PROPERTIES LABELS "validation")

# test that compares the physics output of a full AdePT run against a high-statistics Geant4 simulation using G4HepEm.
# The energy deposition per layer error must be < 1% to pass the test
add_test(NAME testEm3_validation
    COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/validation_testem3.sh
    "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
set_tests_properties(testEm3_validation PROPERTIES LABELS "validation")

# test that compares the physics output of a full AdePT run with transport only in some regions against a high-statistics 
# Geant4 simulation using G4HepEm. The energy deposition per layer error must be < 1% to pass the test
add_test(NAME testEm3_validation_regions
    COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/validation_testem3_regions.sh
    "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
set_tests_properties(testEm3_validation_regions PROPERTIES LABELS "validation")

# Same test as above but using Woodcock tracking in the testEm3 geometry
add_test(NAME testEm3_validation_WDT
  COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/validation_testem3_WDT.sh
  "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
set_tests_properties(testEm3_validation_WDT PROPERTIES LABELS "validation")

# Testing all UI commands of AdePT
add_test(NAME test_UI_commands
  COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/test_UI_commands.sh
  "$<TARGET_FILE:integrationTest>" "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${TEMP_DIR}"
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
set_tests_properties(test_UI_commands PROPERTIES LABELS "unit")

set(PHYSICS_DRIFT_SCENARIOS
  default
  regions
  wdt
  bfield
)

foreach(SCENARIO IN LISTS PHYSICS_DRIFT_SCENARIOS)
  add_test(NAME physics_drift_${SCENARIO}
      COMMAND bash ${PROJECT_SOURCE_DIR}/test/regression/scripts/validation_master_vs_pr.sh
      "$<TARGET_FILE:integrationTest>" "${PROJECT_SOURCE_DIR}" "${SCRIPTS_DIR}" "${PROJECT_BINARY_DIR}/physics_drift_tmp_${SCENARIO}" "${SCENARIO}"
      WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
  set_tests_properties(physics_drift_${SCENARIO} PROPERTIES LABELS "drift;physics_drift")
endforeach()

set(APP_TESTS

)
