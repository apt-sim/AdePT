# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

add_executable(fieldStep field_step.cu)
target_link_libraries(fieldStep AdePT CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml)
set_target_properties(fieldStep PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

