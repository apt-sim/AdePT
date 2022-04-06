# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: Apache-2.0

find_package(HepMC3 QUIET)

if(HepMC3_FOUND)
  message(STATUS "HepMC3 found ${HEPMC3_INCLUDE_DIR}") 
  add_definitions(-DHEPMC3_FOUND)
endif()


if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example14 (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example14 (needs Geant4 with GDML support)")
    return()
  endif()
  if(NOT Geant4_flush_FOUND)
    message(STATUS "Disabling example14 (needs G4VFastSimulationModel::Flush interface available starting with geant4-11-00-ref-01)")
    return()
  endif()
else()
  message(STATUS "Disabling example14 (needs Geant4)")
  return()
endif()

set(sources_g4
  src/ActionInitialisation.cc
  src/DetectorConstruction.cc
  src/DetectorMessenger.cc
  src/EMShowerMessenger.cc
  src/EMShowerModel.cc
  src/EventAction.cc
  src/EventActionMessenger.cc
  src/SimpleHit.cc
  src/PrimaryGeneratorAction.cc
  src/PrimaryGeneratorMessenger.cc
  src/RunAction.cc
  src/SensitiveDetector.cc
  src/PhysListHepEm.cc
  src/TrackingAction.cc
  src/FTFP_BERT_HepEm.cc
)

if(HepMC3_FOUND)
  set(sources_hepmc3
      src/HepMC3G4AsciiReader.cc
      src/HepMC3G4AsciiReaderMessenger.cc
      src/HepMC3G4Interface.cc
     )
endif()

# example14
add_executable(example14 example14.cpp AdeptIntegration.cpp AdeptIntegration.cu BasicScoring.cu 
               ${sources_g4} ${sources_hepmc3})

target_include_directories(example14 PRIVATE ${PROJECT_SOURCE_DIR}/examples/Example14/include 
                          ${PROJECT_SOURCE_DIR}/examples/Example14 ${HEPMC3_INCLUDE_DIR})

target_link_libraries(example14
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

set_target_properties(example14 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Install macros and geometry file

set(TESTEM3_GDML ${CMAKE_BINARY_DIR}/testEm3.gdml)
set(LHCB_GDML ${CMAKE_BINARY_DIR}/LHCb_Upgrade_fullLHCb.gdml)
configure_file("macros/testEm3.gdml" "${CMAKE_BINARY_DIR}/testEm3.gdml")
configure_file("macros/testEm3.mac.in" "${CMAKE_BINARY_DIR}/testEm3.mac")
configure_file("macros/testEm3G4.mac.in" "${CMAKE_BINARY_DIR}/testEm3G4.mac")
configure_file("macros/example14.mac.in" "${CMAKE_BINARY_DIR}/example14.mac")
configure_file("macros/LHCb_Upgrade_fullLHCb.gdml" "${CMAKE_BINARY_DIR}/LHCb_Upgrade_fullLHCb.gdml")
configure_file("macros/lhcb.mac.in" "${CMAKE_BINARY_DIR}/lhcb.mac")

# Tests
add_test(NAME example14-adept-em3
  COMMAND $<TARGET_FILE:example14> -m ${CMAKE_BINARY_DIR}/testEm3.mac)
add_test(NAME example14-g4-em3
  COMMAND $<TARGET_FILE:example14> -m ${CMAKE_BINARY_DIR}/testEm3G4.mac)
