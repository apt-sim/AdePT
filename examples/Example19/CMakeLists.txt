# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: Apache-2.0

set(TargetName SplitKernels)

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling ${TargetName} (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example18 (needs Geant4 with GDML support)")
    return()
  endif()
else()
  message(STATUS "Disabling example18 (needs Geant4)")
  return()
endif()

add_executable(${TargetName}
  main.cpp
  main.cu
  electrons.cu
  gammas.cu)
target_link_libraries(${TargetName}
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

set_target_properties(${TargetName} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# NVTX
target_link_libraries(${TargetName} PRIVATE NVTX)

# Tests
add_test(NAME ${TargetName}
  COMMAND $<TARGET_FILE:${TargetName}> -gdml_file ${CMAKE_BINARY_DIR}/cms2018.gdml)

