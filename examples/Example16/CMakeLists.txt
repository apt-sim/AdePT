# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example16 (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example16 (needs Geant4 with GDML support)")
    return()
  endif()
else()
  message(STATUS "Disabling example16 (needs Geant4)")
  return()
endif()

# example16 is the AdePT demo example using material cuts as defined in the input gdml file
add_executable(example16
  example16.cpp
  example16.cu
  electrons.cu
  gammas.cu)
target_link_libraries(example16
  PRIVATE
    AdePT
    CopCore::CopCore
    VecGeom::vecgeom
    VecGeom::vecgeomcuda_static
    VecGeom::vgdml
    ${Geant4_LIBRARIES}
    G4HepEm::g4HepEmData
    G4HepEm::g4HepEmInit
    G4HepEm::g4HepEmRun
    CUDA::cudart)

set_target_properties(example16 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# NVTX
target_link_libraries(example16 PRIVATE NVTX)

# Tests
add_test(NAME example16
  COMMAND $<TARGET_FILE:example16> -gdml_file ${CMAKE_BINARY_DIR}/cms2018.gdml)
