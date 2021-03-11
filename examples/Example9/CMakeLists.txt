# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example9 (needs G4HepEm)")
  return()
endif()

# Example 9 of particle transportation with GPUs: Electrons and gammas are
# stored in separate containers with monotonic slot numbers, slots are not
# reused. The example includes geometry, magnetic field with constant Bz, and
# physics processes for e-/e+ as well as gammas. Results are reproducible using
# one RANLUX++ state per track.
add_executable(example9 example9.cpp example9.cu electrons.cu gammas.cu relocation.cu)
target_link_libraries(example9 PRIVATE AdePT CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun)
target_compile_options(example9 PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(example9 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
