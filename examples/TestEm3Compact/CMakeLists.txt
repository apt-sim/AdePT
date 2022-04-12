# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling TestEm3 (needs G4HepEm)")
  return()
endif()

# TestEm3 for validation of particle transportation with GPUs.
add_executable(TestEm3-compact TestEm3.cpp TestEm3.cu electrons.cu gammas.cu)
target_link_libraries(TestEm3-compact PRIVATE AdePT CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun)
set_target_properties(TestEm3-compact PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME TestEm3-compact COMMAND TestEm3-compact)
