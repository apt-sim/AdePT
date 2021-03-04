# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Example 2 of particle processing with GPU, including geometry
# and reproducible results using one RANLUX++ state per track.
add_executable(example2 example2.cpp example2.cu)
target_link_libraries(example2 PRIVATE AdePT VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml)
target_compile_options(example2 PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(example2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(BUILD_TESTING)
  file(DOWNLOAD https://gitlab.cern.ch/VecGeom/VecGeom/raw/master/persistency/gdml/gdmls/trackML.gdml trackML.gdml)
  add_test(NAME example2 COMMAND example2 -gdml_name ${CMAKE_BINARY_DIR}/trackML.gdml)
endif()
