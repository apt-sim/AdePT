# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

SET(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example1 (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example1 (needs Geant4 with GDML support)")
    return()
  endif()
  if(NOT Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
    message(STATUS "Disabling example1 (needs G4VTrackingManger interface available starting with geant4-11-00)")
    return()
  endif()
else()
  message(STATUS "Disabling example1 (needs Geant4)")
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
  src/TrackingAction.cc
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
target_link_libraries(example1
  PRIVATE
    ${HEPMC3_LIBRARIES}
    ${HEPMC3_FIO_LIBRARIES}
    AdePT_G4_integration
)

# Install macros and geometry file
SET(GDML ${PROJECT_BINARY_DIR}/cms2018_sd.gdml)
configure_file("macros/example1.mac.in" "${PROJECT_BINARY_DIR}/example1.mac")
configure_file("macros/example1_large_stack.mac.in" "${PROJECT_BINARY_DIR}/example1_large_stack.mac")
configure_file("macros/example1_ttbar.mac.in" "${PROJECT_BINARY_DIR}/example1_ttbar.mac")
configure_file("macros/example1_ttbar_LHCb.mac.in" "${PROJECT_BINARY_DIR}/example1_ttbar_LHCb.mac")
configure_file("macros/example1_ttbar_noadept.mac.in" "${PROJECT_BINARY_DIR}/example1_ttbar_noadept.mac")
configure_file("macros/async_mode.mac.in" "${PROJECT_BINARY_DIR}/async_mode.mac")

# Tests

add_test(NAME example1
  COMMAND $<TARGET_FILE:example1> -m ${PROJECT_BINARY_DIR}/example1_large_stack.mac
)