# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

SET(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling testField (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling testField (needs Geant4 with GDML support)")
    return()
  endif()
else()
  message(STATUS "Disabling testField (needs Geant4)")
  return()
endif()

# testField is the AdePT demo example using material cuts as defined in the input gdml file
add_executable(testField
  testField.cpp
  testField.cu
  electrons.cu
  gammas.cu
)
target_include_directories(testField 
  PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/>
    $<BUILD_INTERFACE:${Geant4_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${G4HepEm_INCLUDE_DIR}>
)
target_link_libraries(testField
  PRIVATE
    CopCore
    VecGeom::vecgeom
    VecGeom::vecgeomcuda_static
    VecGeom::vgdml
    ${Geant4_LIBRARIES}
    ${G4HepEm_LIBRARIES}
    CUDA::cudart
)
if(ADEPT_USE_EXT_BFIELD)
  # link covfie if external magnetic field usage is enabled
  target_link_libraries(testField 
    PRIVATE
      covfie::core
      covfie::cuda
  )
endif()

set_target_properties(testField 
  PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Tests
add_test(NAME testField
  COMMAND $<TARGET_FILE:testField> -gdml_file ${PROJECT_BINARY_DIR}/cms2018.gdml
)
