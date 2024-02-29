# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling TestEm3 (needs G4HepEm)")
  return()
endif()

# TestEm3 for validation of particle transportation with GPUs.
add_executable(TestEm3 TestEm3.cu electrons.cu gammas.cu TestEm3.cpp)
cuda_rdc_target_include_directories(TestEm3 PRIVATE ${PROJECT_SOURCE_DIR}/include)
    # NB: We do not use PRIVATE for executables due to an apparent limitation in CudaRDCUtils
cuda_rdc_target_link_libraries(TestEm3 
  CopCore 
  G4HepEm::g4HepEm
  VecGeom::vecgeom
)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

add_test(NAME TestEm3 COMMAND TestEm3)
