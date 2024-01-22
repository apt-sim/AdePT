# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example12 (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example12 (needs Geant4 with GDML support)")
    return()
  endif()
else()
  message(STATUS "Disabling example12 (needs Geant4)")
  return()
endif()

# example12 is the AdePT demo example using material cuts as defined in the input gdml file
add_executable(example12
  example12.cpp
  example12.cu
  electrons.cu
  gammas.cu)
target_link_libraries(example12 
  PRIVATE
    AdePT
    AdePT_cuda
    CopCore::CopCore
    VecGeom::vecgeom
    VecGeom::vecgeomcuda_static
    VecGeom::vgdml
    ${Geant4_LIBRARIES}
    G4HepEm::g4HepEmData
    G4HepEm::g4HepEmInit
    G4HepEm::g4HepEmRun
    CUDA::cudart)

set_target_properties(example12 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Tests
add_test(NAME example12
  COMMAND $<TARGET_FILE:example12> -gdml_file ${PROJECT_BINARY_DIR}/cms2018.gdml)
