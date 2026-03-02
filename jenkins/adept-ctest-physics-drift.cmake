# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

################################################################################
# CTest dashboard script for physics_drift-only tests.
# Expected environment variables:
# CMAKE_SOURCE_DIR
# CMAKE_BINARY_DIR
# MODEL
# COMPILER
# BUILDTYPE
# PHYSICS_DRIFT_CONFIG
################################################################################

# Build name settings ----------------------------------------------------------
set(CTEST_BUILD_NAME "AdePT-$ENV{COMPILER}-$ENV{BUILDTYPE}")
if(DEFINED ENV{PHYSICS_DRIFT_CONFIG} AND NOT "$ENV{PHYSICS_DRIFT_CONFIG}" STREQUAL "")
  set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME}-physics_drift-$ENV{PHYSICS_DRIFT_CONFIG}")
endif()
if(NOT "$ENV{ghprbPullAuthorLogin}$ENV{ghprbPullId}" STREQUAL "")
  set(CTEST_BUILD_NAME "$ENV{ghprbPullAuthorLogin}#$ENV{ghprbPullId}-${CTEST_BUILD_NAME}")
endif()

# Site name -------------------------------------------------------------------
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

# Cdash Model -----------------------------------------------------------------
if("$ENV{MODEL}" STREQUAL "")
  set(CTEST_MODEL Experimental)
else()
  set(CTEST_MODEL "$ENV{MODEL}")
endif()

set(CTEST_SOURCE_DIRECTORY "$ENV{CMAKE_SOURCE_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{CMAKE_BINARY_DIR}")

include("${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake")

set(ENV{CTEST_OUTPUT_ON_FAILURE} 1)

ctest_start(${CTEST_MODEL} TRACK ${CTEST_MODEL})
ctest_test(BUILD ${CTEST_BINARY_DIRECTORY}
           INCLUDE_LABEL drift
           RETURN_VALUE test_result
           APPEND)
ctest_submit(PARTS Test)

if(test_result)
  message(FATAL_ERROR "physics_drift tests failed")
endif()
