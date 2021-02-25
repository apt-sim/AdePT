# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT Geant4_FOUND)
  message(STATUS "Disabling G4HepEm (needs Geant4)")
  return()
elseif(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/g4hepem/CMakeLists.txt)
  message(STATUS "G4HepEm not found, please run $ git submodule update --init !")
  return()
else()
  set(G4HepEm_CUDA_BUILD ON)
  set(BUILD_TESTING OFF)
  add_subdirectory(g4hepem)
endif()
