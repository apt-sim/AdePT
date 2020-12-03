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
set(TARGET_DEVICE CUDA CACHE STRING "Target architecture of the device")
set_property(CACHE TARGET_DEVICE PROPERTY STRINGS CPU CUDA)
set(TARGET_DEFINITION "TARGET_DEVICE_${TARGET_DEVICE}")

# Enable selected Backend
if(TARGET_DEVICE STREQUAL "CUDA")
  enable_language(CUDA)
  find_package(CUDAToolkit 10.0 REQUIRED)
endif()

# Core library
add_library(CopCore STATIC src/CPUBackend.cpp)
target_compile_definitions(CopCore PUBLIC ${TARGET_DEFINITION})
target_compile_features(CopCore INTERFACE cxx_std_14)
target_include_directories(CopCore INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(CopCore PUBLIC $<$<STREQUAL:"${TARGET_DEVICE}","CUDA">:CUDA::cudart_static>)

# Alias to assist in use as a subproject
add_library(CopCore::CopCore ALIAS CopCore)

# Install headers and target(s)
install(DIRECTORY include/CopCore DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS CopCore EXPORT CopCoreTargets)

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
