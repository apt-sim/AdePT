# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

SET(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling integrationBenchmark (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling integrationBenchmark (needs Geant4 with GDML support)")
    return()
  endif()
  if(NOT Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
    message(STATUS "Disabling integrationBenchmark (needs G4VTrackingManger interface available starting with geant4-11-00)")
    return()
  endif()
else()
  message(STATUS "Disabling integrationBenchmark (needs Geant4)")
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
  src/Run.cc
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

# integrationBenchmark
add_executable(integrationBenchmark integrationBenchmark.cpp ${sources_g4} ${sources_hepmc3})
target_include_directories(integrationBenchmark
  PRIVATE 
    ${PROJECT_SOURCE_DIR}/examples/IntegrationBenchmark/include 
    ${PROJECT_SOURCE_DIR}/examples/IntegrationBenchmark
    ${PROJECT_SOURCE_DIR}/examples/common/include
    ${HEPMC3_INCLUDE_DIR}
)
target_link_libraries(integrationBenchmark
  PRIVATE
    AdePT_G4_integration  
    ${HEPMC3_LIBRARIES} 
    ${HEPMC3_FIO_LIBRARIES}
)

# Scripts
configure_file("macros/integrationbenchmark.mac.in" "${CMAKE_BINARY_DIR}/integrationbenchmark.mac")

# Tests
add_test(NAME integrationBenchmark
  COMMAND $<TARGET_FILE:integrationBenchmark> -m ${PROJECT_BINARY_DIR}/example1_large_stack.mac
)