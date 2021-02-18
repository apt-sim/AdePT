# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0


set(ADEPT_COMMON_SRCS
  Raytracer.cpp
)

# add Raytracer.cu
foreach(SRC_FILE ${ADEPT_COMMON_SRCS})
  get_filename_component(SRC_FILENAME ${SRC_FILE} NAME_WE)

  set(SRC_FILE_CPP ${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE})
  set(SRC_FILE_CU ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${SRC_FILENAME}.cu)

  add_custom_command(
    OUTPUT ${SRC_FILE_CU}
    COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE_CPP} ${SRC_FILE_CU}
    DEPENDS ${SRC_FILE_CPP})

  list(APPEND ADEPT_CUDA_SRCS ${SRC_FILE_CU})

endforeach()

list(APPEND ADEPT_CUDA_SRCS
  RaytraceBenchmark.cu
)

# library used for RaytraceBenchmark.cpp
add_library(raytracercu ${ADEPT_CUDA_SRCS})
target_include_directories(raytracercu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(raytracercu AdePT CopCore::CopCore VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static)
target_compile_options(raytracercu PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(raytracercu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#exec RaytraceBenchmark.cpp
add_executable(RaytraceBenchmark Raytracer.cpp RaytraceBenchmark.cpp)
target_link_libraries(RaytraceBenchmark PRIVATE raytracercu AdePT CopCore::CopCore VecGeom::vecgeom VecGeom::vecgeomcuda VecGeom::vgdml)
target_compile_options(RaytraceBenchmark PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(RaytraceBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

