# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

set(ADEPT_COMMON_SRCS
  field_step.cu
)

# add field_step.cu
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
#	field_step.cu
)

# library used for fieldStepExe.cpp
#add_library(field_step_culib ${ADEPT_CUDA_SRCS})
#target_include_directories(field_step_culib PUBLIC
#  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
#  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
#  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
#  $<INSTALL_INTERFACE:base>  
#  )
# target_link_libraries(field_step_culib VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda_static CopCore::CopCore)
# target_compile_options(field_step_culib PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# set_target_properties(field_step_culib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#exec fieldStepExe
add_executable(fieldStepExe field_step.cu)
target_include_directories(fieldStepExe PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(fieldStepExe VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda VecGeom::vgdml # field_step_culib
                                   CopCore::CopCore)
target_compile_options(fieldStepExe PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(fieldStepExe PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

