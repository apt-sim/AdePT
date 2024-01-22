# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example6 (needs G4HepEm)")
  return()
endif()

add_executable(example6 example6.cpp example6.cu)
target_link_libraries(example6 PRIVATE AdePT AdePT_cuda CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun)
set_target_properties(example6 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME example6 COMMAND example6 -gdml_file "${TESTING_GDML}")
