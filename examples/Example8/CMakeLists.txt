# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example8 (needs G4HepEm)")
  return()
endif()

# Example 8 of particle transportation with GPUs: Tracks are stored with a
# monotonic slot number, slots are not reused. The example includes geometry,
# magnetic field with constant Bz, and physics processes for e-/e+. Photons
# immediately pair-produce if allowed or deposit their energy. Results are
# reproducible using one RANLUX++ state per track.
add_executable(example8 example8.cpp example8.cu)
target_link_libraries(example8 PRIVATE AdePT CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun)
target_compile_options(example8 PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(example8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
