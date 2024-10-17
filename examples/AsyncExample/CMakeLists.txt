# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

if(HepMC3_FOUND)
  message(STATUS "HepMC3 found ${HEPMC3_INCLUDE_DIR}")
else()
  find_package(HepMC3)
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
  src/EventAction.cc
  src/EventActionMessenger.cc
  src/SimpleHit.cc
  src/RunAction.cc
  src/SensitiveDetector.cc
  src/TrackingAction.cc
)

# Make use of common source shared between examples:
add_library(ExamplesCommon OBJECT
    ../common/src/ParticleGun.cc
    ../common/src/ParticleGunMessenger.cc
    ../common/src/FTFP_BERT_HepEm.cc
    ../common/src/FTFP_BERT_AdePT.cc
)

target_include_directories(ExamplesCommon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../common/include)
target_link_libraries(ExamplesCommon PUBLIC AdePT_G4_integration)
target_compile_options(ExamplesCommon PRIVATE -Wall -Wextra)

if(HepMC3_FOUND)
  target_sources(ExamplesCommon PRIVATE
    ../common/src/HepMC3G4AsciiReader.cc
    ../common/src/HepMC3G4AsciiReaderMessenger.cc
    ../common/src/HepMC3G4Interface.cc
  )
  target_link_libraries(ExamplesCommon PRIVATE HepMC3::HepMC3)
endif()

add_library(AdePTAsyncTransport SHARED
  AdeptIntegration.cpp
  AdeptIntegration.cu
  BasicScoring.cu
  ResourceManagement.cu)
target_link_libraries(AdePTAsyncTransport
  PUBLIC
    AdePT_G4_integration
    VecGeom::vecgeomcuda_static
    )
set_target_properties(AdePTAsyncTransport PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

option(AsyncTransport "Link to AdePTAsyncTransport" ON)

find_package(ROOT REQUIRED COMPONENTS Hist RIO)

add_executable(AsyncExample
    example.cpp
    ${sources_g4})
target_include_directories(AsyncExample PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(AsyncExample
  PRIVATE
    ExamplesCommon
    NVTX
    ROOT::Core
    ROOT::Hist
    ROOT::RIO
)

if(AsyncTransport)
  target_link_libraries(AsyncExample PRIVATE AdePTAsyncTransport)
else()
  target_link_libraries(AsyncExample PRIVATE AdePTTransport)
endif()

target_compile_options(AsyncExample PRIVATE -Wall -Wextra)

# Install macros and geometry file
configure_file("macros/AsyncExample.mac.in" "${CMAKE_BINARY_DIR}/AsyncExample.mac")
