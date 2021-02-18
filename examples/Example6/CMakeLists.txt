# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET g4HepEm)
  message(STATUS "Disabling example6 (needs G4HepEm)")
  return()
endif()

# Example 6 of particle transportation with GPUs: one container for all tracks,
# geometry, magnetic field with constant Bz, and physics processes for e-/e+.
# Photons immediately pair-produce if allowed or deposit their energy. Results
# are reproducible using one RANLUX++ state per track.
add_executable(example6 example6.cpp example6.cu)
target_link_libraries(example6 PRIVATE AdePT CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml ${Geant4_LIBRARIES} g4HepEmData g4HepEmInit g4HepEmRun)
target_compile_options(example6 PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(example6 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
