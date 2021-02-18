# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example5 (needs G4HepEm)")
  return()
endif()

add_executable(example5 example5.cu)
target_link_libraries(example5 PRIVATE ${Geant4_LIBRARIES} g4HepEmData g4HepEmInit g4HepEmRun CopCore::CopCore)
