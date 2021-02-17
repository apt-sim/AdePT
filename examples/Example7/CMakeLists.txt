# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET g4HepEm)
  message(STATUS "Disabling example7 (needs G4HepEm)")
  return()
endif()

if(Geant4_FOUND)
  if(NOT Geant4_gdml_FOUND)
    message(STATUS "Disabling example7 (needs Geant4 with GDML support)")
  endif()
else()
  message(STATUS "Disabling example7 (needs Geant4)")
endif()

# Refactor I/O into separate library here...

# example7-geant4 is the "full" Geant4 application for data export
add_executable(example7-geant4 example7-geant4.cc)
target_include_directories(example7-geant4
  PRIVATE
    "${PROJECT_SOURCE_DIR}/base/inc"
    "${PROJECT_SOURCE_DIR}/base/inc/CopCore/include")
target_link_libraries(example7-geant4 PRIVATE ${Geant4_LIBRARIES} G4HepEm::g4HepEm)

# example7-adept is the pure AdePT application for the data import
add_executable(example7-adept example7-adept.cc)
target_include_directories(example7-adept
  PRIVATE
    "${PROJECT_SOURCE_DIR}/base/inc"
    "${PROJECT_SOURCE_DIR}/base/inc/CopCore/include")
# Partially copied from example6, but causing link error locally (TBU)
target_link_libraries(example7-adept PRIVATE VecGeom::vgdml)