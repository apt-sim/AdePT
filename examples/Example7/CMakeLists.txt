# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example7 (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example7 (needs Geant4 with GDML support)")
    return()
  endif()
else()
  message(STATUS "Disabling example7 (needs Geant4)")
  return()
endif()

# Also need VecGeom 1.1.14 or newer
if(VecGeom_VERSION VERSION_LESS 1.1.14)
  message(STATUS "Disabling example7 (requires VecGeom 1.1.14 or newer to support full GDML parsing)")
  return()
else()
  if(NOT TARGET VecGeom::vgdml)
    message(STATUS "Disabling example7 (requires VecGeom with GDML support)")
    return()
  endif()
endif()

# example7-geant4 is the "full" Geant4 application for data export
add_executable(example7-geant4 example7-geant4.cc geant4_mock.h geant4_mock.cc)
target_link_libraries(example7-geant4 PRIVATE AdePT ${Geant4_LIBRARIES} G4HepEm::g4HepEm G4HepEm::g4HepEmDataJsonIO)

# example7-adept is the pure AdePT application for the data import
add_executable(example7-adept
  example7-adept.cc
  TestEm3.cu
  electrons.cu
  gammas.cu
  relocation.cu)
target_link_libraries(example7-adept
  PRIVATE
    AdePT
    CopCore::CopCore
    VecGeom::vecgeom
    VecGeom::vecgeomcuda_static
    VecGeom::vgdml
    G4HepEm::g4HepEmData
    G4HepEm::g4HepEmDataJsonIO
    G4HepEm::g4HepEmRun
    CUDA::cudart)
set_target_properties(example7-adept PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Tests
add_test(NAME example7-geant4 COMMAND example7-geant4)
add_test(NAME example7-adept COMMAND example7-adept -gdml_file example7.gdml)
set_tests_properties(example7-adept PROPERTIES DEPENDS "example7-geant4")
