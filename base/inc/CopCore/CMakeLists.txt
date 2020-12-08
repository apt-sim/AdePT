# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(CopCore VERSION 0.1.0)

include(GNUInstallDirs)

# ---------------------
# Device Target Options:
# These can be set from the external project (Namespaced?)
# ---------------------
# Device target architecture
option(COPCORE_ENABLE_CUDA "Build CopCore_CUDA library" ON)
if(COPCORE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 10.0 REQUIRED)
endif()

# CPU library
add_library(CopCore_CPU STATIC src/CPUBackend.cpp)
target_compile_definitions(CopCore_CPU PUBLIC TARGET_DEVICE_CPU)
target_compile_features(CopCore_CPU PUBLIC cxx_std_14)
target_include_directories(CopCore_CPU PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
add_library(CopCore::CopCore_CPU ALIAS CopCore_CPU)

# CUDA library
if(COPCORE_ENABLE_CUDA)
  add_library(CopCore_CUDA INTERFACE)
  target_compile_definitions(CopCore_CUDA INTERFACE TARGET_DEVICE_CUDA)
  target_compile_features(CopCore_CUDA INTERFACE cxx_std_14)
  target_include_directories(CopCore_CUDA INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(CopCore_CUDA INTERFACE CUDA::cudart_static)

  add_library(CopCore::CopCore_CUDA ALIAS CopCore_CUDA)
endif()

# Install headers and target(s)
install(DIRECTORY include/CopCore DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS CopCore_CPU EXPORT CopCoreTargets)

# Support files
include(CMakePackageConfigHelpers)

write_basic_package_version_file("${PROJECT_BINARY_DIR}/CopCoreConfigVersion.cmake"
  COMPATIBILITY AnyNewerVersion
  ARCH_INDEPENDENT)

configure_package_config_file(CopCoreConfig.cmake.in "${PROJECT_BINARY_DIR}/CopCoreConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CopCore"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR)

install(FILES "${PROJECT_BINARY_DIR}/CopCoreConfigVersion.cmake" "${PROJECT_BINARY_DIR}/CopCoreConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CopCore")

install(EXPORT CopCoreTargets
  NAMESPACE CopCore::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CopCore")
