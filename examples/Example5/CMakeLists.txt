# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling example5 (needs G4HepEm)")
  return()
endif()

add_executable(example5 example5.cu)
target_link_libraries(example5 PRIVATE CopCore::CopCore ${Geant4_LIBRARIES} G4HepEm::g4HepEmData G4HepEm::g4HepEmInit G4HepEm::g4HepEmRun)

add_test(NAME example5 COMMAND example5)
