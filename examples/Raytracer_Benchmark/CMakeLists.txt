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
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
)

target_link_libraries(raytracercu VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static CopCore::CopCore)
target_compile_options(raytracercu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-fmad=false>")
set_target_properties(raytracercu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#exec RaytraceBenchmark.cpp
add_executable(RaytraceBenchmark Raytracer.cpp RaytraceBenchmark.cpp)
target_include_directories(RaytraceBenchmark PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(RaytraceBenchmark VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml raytracercu CopCore::CopCore)
target_compile_options(RaytraceBenchmark PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-fmad=false>")
set_target_properties(RaytraceBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME RaytraceBenchmarkCPU
  COMMAND $<TARGET_FILE:RaytraceBenchmark> -gdml_file ${CMAKE_BINARY_DIR}/trackML.gdml -on_gpu 0)
set_tests_properties(RaytraceBenchmarkCPU PROPERTIES ATTACHED_FILES_ON_FAIL "output-cpu.ppm")

add_test(NAME RaytraceBenchmarkGPU
  COMMAND $<TARGET_FILE:RaytraceBenchmark> -gdml_file ${CMAKE_BINARY_DIR}/trackML.gdml -on_gpu 1)
set_tests_properties(RaytraceBenchmarkGPU PROPERTIES ATTACHED_FILES_ON_FAIL "output-gpu.ppm")

add_test(NAME RaytraceBenchmarkDiff
  COMMAND diff output-cpu.ppm output-gpu.ppm)

find_program(COMPARE NAMES compare)

if(EXISTS ${COMPARE})
  add_test(NAME RaytraceBenchmarkCompare
    COMMAND ${COMPARE} -metric MAE output-cpu.ppm output-gpu.ppm diff.ppm)

  set_tests_properties(RaytraceBenchmarkCompare PROPERTIES
    ATTACHED_FILES_ON_FAIL "${CMAKE_CURRENT_BINARY_DIR}/diff.ppm"
    REQUIRED_FILES "output-cpu.ppm;output-gpu.ppm"
    DEPENDS "RaytraceBenchmarkCPU;RaytraceBenchmarkGPU"
    PASS_REGULAR_EXPRESSION "^0 \\(0\\)")
endif()
