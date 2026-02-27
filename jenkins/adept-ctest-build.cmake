# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0
################################################################################
# Build-only CTest dashboard script.
# Expected environment variables:
# CMAKE_SOURCE_DIR        // CMake source directory
# CMAKE_BINARY_DIR        // CMake binary directory
# CMAKE_INSTALL_PREFIX    // Installation prefix for CMake (Jenkins trigger)
# BUILDTYPE               // CMake build type: Debug, Release
# COMPILER                // Compiler keyword: gcc8, clang10
# MODEL                   // CTest model (Experimental, Continuous, or Nightly)
# ExtraCMakeOptions       // Additional CMake configure flags
################################################################################

# Build name settings (CTEST_BUILD_NAME)----------------------------------------
set(CTEST_BUILD_NAME "AdePT-$ENV{COMPILER}-$ENV{BUILDTYPE}")
if(NOT "$ENV{ghprbPullAuthorLogin}$ENV{ghprbPullId}" STREQUAL "")
  set(CTEST_BUILD_NAME "$ENV{ghprbPullAuthorLogin}#$ENV{ghprbPullId}-${CTEST_BUILD_NAME}")
endif()

# Site name (CTEST_SITE)--------------------------------------------------------
if(DEFINED ENV{CTEST_SITE})
  set(CTEST_SITE $ENV{CTEST_SITE})
elseif(DEFINED ENV{container} AND DEFINED ENV{NODE_NAME})
  set(CTEST_SITE "$ENV{NODE_NAME}-$ENV{container}")
else()
  find_program(HOSTNAME_CMD NAMES hostname)
  execute_process(COMMAND ${HOSTNAME_CMD}
                  OUTPUT_VARIABLE CTEST_SITE
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Cdash Model (CTEST_MODEL)-----------------------------------------------------
if("$ENV{MODEL}" STREQUAL "")
  set(CTEST_MODEL Experimental)
else()
  set(CTEST_MODEL "$ENV{MODEL}")
endif()

# Use multiple CPU cores to build-----------------------------------------------
include(ProcessorCount)
ProcessorCount(N)

# CTest/CMake settings----------------------------------------------------------
if(DEFINED ENV{BUILDTYPE})
  set(CTEST_BUILD_CONFIGURATION "$ENV{BUILDTYPE}")
else()
  set(CTEST_BUILD_CONFIGURATION Release)
endif()
if(DEFINED ENV{CMAKE_INSTALL_PREFIX})
  set(CTEST_INSTALL_PREFIX "$ENV{CMAKE_INSTALL_PREFIX}")
else()
  set(CTEST_INSTALL_PREFIX "$ENV{PWD}/install")
endif()
if(DEFINED ENV{CMAKE_SOURCE_DIR})
  set(CTEST_SOURCE_DIRECTORY "$ENV{CMAKE_SOURCE_DIR}")
else()
  get_filename_component(CTEST_SOURCE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
endif()
if(DEFINED ENV{CMAKE_BINARY_DIR})
  set(CTEST_BINARY_DIRECTORY "$ENV{CMAKE_BINARY_DIR}")
else()
  set(CTEST_BINARY_DIRECTORY "$ENV{PWD}/build")
endif()
if(NOT DEFINED ENV{CUDA_CAPABILITY})
  set(ENV{CUDA_CAPABILITY} 75)
endif()
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_FLAGS "-j${N}")

# Split the environment string into a list of individual flags to enable passing multiple arguments to ExtraCMakeOptions
separate_arguments(extra_cmake_args UNIX_COMMAND "$ENV{ExtraCMakeOptions}")

# Fixed set of CMake options----------------------------------------------------
set(config_options -DCMAKE_INSTALL_PREFIX=${CTEST_INSTALL_PREFIX}
                   -DCMAKE_CUDA_ARCHITECTURES=$ENV{CUDA_CAPABILITY}
                   ${extra_cmake_args})

# Print summary information-----------------------------------------------------
foreach(v
    CTEST_SITE
    CTEST_BUILD_NAME
    CTEST_SOURCE_DIRECTORY
    CTEST_BINARY_DIRECTORY
    CTEST_INSTALL_PREFIX
    CTEST_CMAKE_GENERATOR
    )
  set(vars "${vars}  ${v}=[${${v}}]\n")
endforeach(v)
message("Dashboard build script configuration:\n${vars}\n")

include("${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake")

# clean build tree for this dashboard slot--------------------------------------
file(REMOVE_RECURSE ${CTEST_BINARY_DIRECTORY})

ctest_start(${CTEST_MODEL} TRACK ${CTEST_MODEL})

ctest_configure(BUILD   ${CTEST_BINARY_DIRECTORY}
                SOURCE  ${CTEST_SOURCE_DIRECTORY}
                OPTIONS "${config_options}"
                APPEND)
ctest_submit(PARTS Configure Notes)

ctest_build(BUILD ${CTEST_BINARY_DIRECTORY}
            TARGET all
            RETURN_VALUE build_result
            APPEND)
ctest_submit(PARTS Build)

if(build_result)
  message(FATAL_ERROR "Failed to build all targets")
endif()
