# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(CopCore VERSION 0.1.0)

include(GNUInstallDirs)

# Core library
add_library(CopCore INTERFACE)
target_compile_features(CopCore INTERFACE cxx_std_14)
target_include_directories(CopCore INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Alias to assist in use as a subproject
add_library(CopCore::CopCore ALIAS CopCore)

# Install headers and target(s)
install(DIRECTORY include/CopCore 
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS CopCore 
        EXPORT CopCoreTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

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

export(TARGETS CopCore
  NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake)

install(EXPORT CopCoreTargets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CopCore")
