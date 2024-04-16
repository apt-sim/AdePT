# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

find_package(HepMC3 QUIET)

if(HepMC3_FOUND)
  message(STATUS "HepMC3 found ${HEPMC3_INCLUDE_DIR}")
  add_definitions(-DHEPMC3_FOUND)
endif()


if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling AsyncExample (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling AsyncExample (needs Geant4 with GDML support)")
    return()
  endif()
  if(FALSE AND NOT Geant4_flush_FOUND)
    message(STATUS "Disabling AsyncExample (needs G4VFastSimulationModel::Flush interface available starting with geant4-11-00-ref-01)")
    return()
  endif()
else()
  message(STATUS "Disabling AsyncExample (needs Geant4)")
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
  src/ParticleGun.cc
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

add_executable(AsyncExample example.cpp AdeptIntegration.cpp AdeptIntegration.cu BasicScoring.cu ResourceManagement.cu
               ${sources_g4} ${sources_hepmc3})

target_include_directories(AsyncExample PRIVATE ${PROJECT_SOURCE_DIR}/examples/AsyncExample/include
  ${PROJECT_SOURCE_DIR}/examples/AsyncExample ${HEPMC3_INCLUDE_DIR})

target_link_libraries(AsyncExample
  PRIVATE
    AdePT_G4_integration
    CUDA::cudart
    ${HEPMC3_LIBRARIES} ${HEPMC3_FIO_LIBRARIES}
    NVTX)

set_target_properties(AsyncExample PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_options(AsyncExample PRIVATE -Wall -Wextra)
# g4HepEm headers generate warnings, so declare them as system headers:

# Install macros and geometry file

configure_file("macros/AsyncExample.mac.in" "${CMAKE_BINARY_DIR}/AsyncExample.mac")

