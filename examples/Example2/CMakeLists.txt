# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Example 2 of particle processing with GPU, including geometry
# and reproducible results using one RANLUX++ state per track.
add_executable(example2 example2.cpp example2.cu)
target_link_libraries(example2 PRIVATE AdePT VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml)
set_target_properties(example2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME example2 COMMAND example2 -gdml_name "${TESTING_GDML}")
