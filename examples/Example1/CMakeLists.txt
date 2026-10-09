# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

SET(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example1 (needs G4HepEm)")
  return()
endif()

if(NOT Geant4_gdml_FOUND)
  message(STATUS "Disabling example1 (needs Geant4 with GDML support)")
  return()
endif()
if(NOT AdePT_HAS_G4VTRACKINGMANAGER)
  message(STATUS "Disabling example1 (needs G4VTrackingManger interface available starting with geant4-11-00)")
  return()
endif()

set(sources_g4
  ${PROJECT_SOURCE_DIR}/examples/common/src/ParticleGun.cc
  ${PROJECT_SOURCE_DIR}/examples/common/src/ParticleGunMessenger.cc
  ${PROJECT_SOURCE_DIR}/examples/common/src/FTFP_BERT_HepEm.cc
  ${PROJECT_SOURCE_DIR}/examples/common/src/FTFP_BERT_AdePT.cc
  src/ActionInitialisation.cc
  src/DetectorConstruction.cc
  src/DetectorMessenger.cc
  src/EventAction.cc
  src/EventActionMessenger.cc
  src/SteppingAction.cc
  src/SimpleHit.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/SensitiveDetector.cc
)

if(HepMC3_FOUND)
  set(sources_hepmc3
    ${PROJECT_SOURCE_DIR}/examples/common/src/HepMC3G4AsciiReader.cc
    ${PROJECT_SOURCE_DIR}/examples/common/src/HepMC3G4AsciiReaderMessenger.cc
    ${PROJECT_SOURCE_DIR}/examples/common/src/HepMC3G4Interface.cc
  )
endif()

# example1
add_executable(example1 example1.cpp ${sources_g4} ${sources_hepmc3})
target_include_directories(example1
  PRIVATE
    ${PROJECT_SOURCE_DIR}/examples/Example1/include
    ${PROJECT_SOURCE_DIR}/examples/Example1
    ${PROJECT_SOURCE_DIR}/examples/common/include
    ${HEPMC3_INCLUDE_DIR}
)
cuda_rdc_target_link_libraries(example1
  PRIVATE
    ${HEPMC3_LIBRARIES}
    ${HEPMC3_FIO_LIBRARIES}
    AdePT::G4Integration
)

# Install macros and geometry file
SET(GDML ${PROJECT_BINARY_DIR}/cms2018_sd.gdml)
configure_file("macros/example1.mac.in" "${PROJECT_BINARY_DIR}/example1.mac")
configure_file("macros/example1_ttbar.mac.in" "${PROJECT_BINARY_DIR}/example1_ttbar.mac")
configure_file("macros/example1_ttbar_LHCb.mac.in" "${PROJECT_BINARY_DIR}/example1_ttbar_LHCb.mac")
configure_file("macros/example1_ttbar_noadept.mac.in" "${PROJECT_BINARY_DIR}/example1_ttbar_noadept.mac")

# Tests

add_test(NAME example1-g4vg
  COMMAND $<TARGET_FILE:example1> -m ${PROJECT_BINARY_DIR}/example1.mac
)

if(ADEPT_GEOMETRY_VALIDATION)
  # Geometry validation of the GPU -> Geant4 handoff: only some regions are transported on the GPU and every track leaving them is
  # checked (/adept/validateHandoff). The default case uses the CMS 2018 geometry with its ECAL region on the GPU.
  set(ADEPT_HANDOFF_VALIDATION_CUDA_STACK 8192 CACHE STRING
      "CUDA stack limit of the example1 handoff validation tests (deep Boolean solids may need more)")

  function(add_example1_handoff_test name gdml regions)
    set(HANDOFF_GDML ${gdml})
    set(HANDOFF_CUDA_STACK ${ADEPT_HANDOFF_VALIDATION_CUDA_STACK})
    set(HANDOFF_RECORDS ${PROJECT_BINARY_DIR}/example1_${name}_crossings.csv)
    set(HANDOFF_GPU_REGIONS "")
    foreach(region IN LISTS regions)
      string(APPEND HANDOFF_GPU_REGIONS "/adept/addGPURegion ${region}\n")
    endforeach()
    configure_file("macros/example1_handoff.mac.in" "${PROJECT_BINARY_DIR}/example1_${name}.mac")
    add_test(NAME example1-${name} COMMAND $<TARGET_FILE:example1> -m ${PROJECT_BINARY_DIR}/example1_${name}.mac)
    set_tests_properties(example1-${name} PROPERTIES PASS_REGULAR_EXPRESSION "HandoffValidation: PASSED"
                                                     FAIL_REGULAR_EXPRESSION "HandoffValidation: FAILED")
  endfunction()

  add_example1_handoff_test(handoff ${GDML} "EcalRegion")

  # External geometries defining their own regions, e.g.
  #   -DADEPT_HANDOFF_VALIDATION_GEOMETRIES="/path/cmsD110.gdml=EcalRegion,HcalRegion;/path/other.gdml=SomeRegion"
  # Each entry adds a test example1-handoff-<file name> with the listed regions on the GPU.
  set(ADEPT_HANDOFF_VALIDATION_GEOMETRIES "" CACHE STRING
      "List of <gdml file>=<region>[,<region>...] entries for additional handoff validation tests of example1")
  foreach(entry IN LISTS ADEPT_HANDOFF_VALIDATION_GEOMETRIES)
    string(FIND "${entry}" "=" separator REVERSE)
    if(separator LESS 1)
      message(FATAL_ERROR "ADEPT_HANDOFF_VALIDATION_GEOMETRIES entry '${entry}' is not <gdml file>=<region>[,...]")
    endif()
    string(SUBSTRING "${entry}" 0 ${separator} gdml)
    math(EXPR first "${separator} + 1")
    string(SUBSTRING "${entry}" ${first} -1 regions)
    string(REPLACE "," ";" regions "${regions}")
    get_filename_component(gdmlName ${gdml} NAME_WE)
    add_example1_handoff_test(handoff-${gdmlName} ${gdml} "${regions}")
  endforeach()
endif()
