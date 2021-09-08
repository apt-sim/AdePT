# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

option(BUILTIN_G4VECGEOMNAV "Build G4VecGeomNav from source (requires VecGeom)" ON)

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example12 (needs G4HepEm)")
  return()
endif()

# Start  Configuration of G4VecGeomNav - converter from G4 to VecGeom volumes
# -----  -------------    ------------
# Minimum version of G4VecGeomNav we need.
set(G4VecGeomNav_VERSION "") # 2021.06.18") # 0.1")

if(BUILTIN_G4VECGEOMNAV)
  include(FetchContent)

  FetchContent_Declare(
    g4vecgeomnav
    GIT_REPOSITORY https://gitlab.cern.ch/VecGeom/g4vecgeomnav.git
    GIT_TAG 1a002fbf2bccd81827b6100f86e2ad82433a75e9
  )

  FetchContent_GetProperties(g4vecgeomnav)

  if(NOT g4vecgeomnav_POPULATED)
    FetchContent_Populate(g4vecgeomnav)
    message(STATUS "${g4vecgeomnav_SOURCE_DIR}, ${g4vecgeomnav_BINARY_DIR}")
    add_subdirectory(${g4vecgeomnav_SOURCE_DIR} ${g4vecgeomnav_BINARY_DIR})
  endif()
else()
  find_package(G4VecGeomNav REQUIRED)
endif()
# End   configuration of G4VecGeomNav 
# ---   -------------    ------------

add_executable(example12 example12.cpp example12.cu electrons.cu gammas.cu)
target_link_libraries(example12
   PRIVATE
     AdePT CopCore::CopCore
     G4VecGeomNav::g4vecgeomnav
     VecGeom::vecgeom VecGeom::vecgeomcuda_static VecGeom::vgdml
     ${Geant4_LIBRARIES}
     G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun)
set_target_properties(example12 PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME example12 COMMAND example12 -gdml_name "${TESTING_GDML}")
