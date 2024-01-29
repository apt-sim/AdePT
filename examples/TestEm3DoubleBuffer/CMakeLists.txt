# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling TestEm3 (needs G4HepEm)")
  return()
endif()

add_executable(TestEm3DoubleBuffer TestEm3.cpp TestEm3.cu electrons.cu gammas.cu)
target_link_libraries(TestEm3DoubleBuffer PRIVATE AdePT_G4_integration AdePT AdePT_cuda CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun)
set_target_properties(TestEm3DoubleBuffer PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME TestEm3DoubleBuffer COMMAND TestEm3DoubleBuffer)
