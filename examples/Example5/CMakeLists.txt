# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET g4HepEm)
  message(STATUS "Disabling example5 (needs G4HepEm)")
  return()
endif()

add_executable(example5 example5.cu)
target_include_directories(example5 PRIVATE
  $<TARGET_PROPERTY:g4HepEmRun,INCLUDE_DIRECTORIES>
)
target_compile_definitions(example5 PRIVATE G4HepEm_CUDA_BUILD)
target_link_libraries(example5 PRIVATE ${Geant4_LIBRARIES} g4HepEmData g4HepEmInit CopCore::CopCore)
