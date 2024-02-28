# SPDX-FileCopyrightText: 2021 CERN
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET G4HepEm::g4HepEm)
  message(STATUS "Disabling TestEm3 (needs G4HepEm)")
  return()
endif()

# TestEm3 for validation of particle transportation with GPUs.
add_executable(TestEm3 TestEm3.cu electrons.cu gammas.cu TestEm3.cpp)
target_include_directories(TestEm3 PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(TestEm3 
  PRIVATE 
    CopCore 
    AdePT_G4_integration
    AdePT 
    AdePT_cuda_standalone 
    VecGeom::vecgeomcuda_static
)
set_target_properties(TestEm3 
  PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

add_test(NAME TestEm3 COMMAND TestEm3)
