# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Example 4 of particle processing with GPU, including geometry, magnetic field
# and reproducible results using one RANLUX++ state per track.
add_executable(example4 example4.cpp example4.cu)
target_link_libraries(example4 PRIVATE AdePT CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml)
set_target_properties(example4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME example4 COMMAND example4 -gdml_name "${TESTING_GDML}")
