# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example21 (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example21 (needs Geant4 with GDML support)")
    return()
  endif()
  if(NOT Geant4_VERSION VERSION_GREATER_EQUAL 11.0)
    message(STATUS "Disabling example21 (needs G4VTrackingManger interface available starting with geant4-11-00)")
    return()
  endif()
else()
  message(STATUS "Disabling example21 (needs Geant4)")
  return()
endif()

set(sources_g4
  src/AdePTTrackingManager.cc
  src/PhysListAdePT.cc
  src/ActionInitialisation.cc
  src/DetectorConstruction.cc
  src/DetectorMessenger.cc
  src/EventAction.cc
  src/EventActionMessenger.cc
  src/SteppingAction.cc
  src/SimpleHit.cc
  src/PrimaryGeneratorAction.cc
  src/PrimaryGeneratorMessenger.cc
  src/RunAction.cc
  src/Run.cc
  src/SensitiveDetector.cc
  src/PhysListHepEm.cc
  src/TrackingAction.cc
  src/FTFP_BERT_HepEm.cc
  src/FTFP_BERT_AdePT.cc
  src/ParticleGun.cc
)

if(HepMC3_FOUND)
  set(sources_hepmc3
      src/HepMC3G4AsciiReader.cc
      src/HepMC3G4AsciiReaderMessenger.cc
      src/HepMC3G4Interface.cc
     )
endif()

# example21
add_executable(example21 example21.cpp AdeptIntegration.cpp AdeptIntegration.cu BasicScoring.cu 
               ${sources_g4} ${sources_hepmc3})

target_include_directories(example21 PRIVATE ${PROJECT_SOURCE_DIR}/examples/Example21/include 
                          ${PROJECT_SOURCE_DIR}/examples/Example21 ${HEPMC3_INCLUDE_DIR}
                          ${PROJECT_SOURCE_DIR}/base/inc/AdePT)

target_link_libraries(example21
  PRIVATE
    AdePT
    CopCore::CopCore
    VecGeom::vecgeom
    VecGeom::vecgeomcuda_static
    VecGeom::vgdml
    ${Geant4_LIBRARIES}
    G4HepEm::g4HepEm
    G4HepEm::g4HepEmData
    G4HepEm::g4HepEmInit
    G4HepEm::g4HepEmRun
    CUDA::cudart
    ${HEPMC3_LIBRARIES} ${HEPMC3_FIO_LIBRARIES})

set_target_properties(example21 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Install macros and geometry file

set(TESTEM3_GDML ${CMAKE_BINARY_DIR}/testEm3.gdml)
set(LHCB_GDML ${CMAKE_BINARY_DIR}/LHCb_Upgrade_fullLHCb.gdml)
configure_file("../data/testEm3.gdml" "${CMAKE_BINARY_DIR}/testEm3.gdml")
configure_file("macros/testEm3.mac.in" "${CMAKE_BINARY_DIR}/testEm3_e17.mac")
configure_file("macros/example21.mac.in" "${CMAKE_BINARY_DIR}/example21.mac")
configure_file("macros/example21_ttbar.mac.in" "${CMAKE_BINARY_DIR}/example21_ttbar.mac")
configure_file("../data/LHCb_Upgrade_fullLHCb.gdml" "${CMAKE_BINARY_DIR}/LHCb_Upgrade_fullLHCb.gdml")
configure_file("macros/lhcb.mac.in" "${CMAKE_BINARY_DIR}/lhcb_e17.mac")
configure_file("macros/lhcb_validation.mac.in" "${CMAKE_BINARY_DIR}/lhcb_validation.mac")

# Tests
add_test(NAME example21-adept-em3
  COMMAND $<TARGET_FILE:example21> -m ${CMAKE_BINARY_DIR}/testEm3_e17.mac)

