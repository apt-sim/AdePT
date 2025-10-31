# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25.2)

# Record the command line invoking the cmake command. Replay with recmake_initial.sh.
include(cmake/RecordCmdLine.cmake)

project(AdePT
  VERSION 0.1.0
  DESCRIPTION "Accelerated demonstrator of electromagnetic Particle Transport"
  LANGUAGES C CXX CUDA
)

#----------------------------------------------------------------------------#
# CMake and Build Settings
#----------------------------------------------------------------------------#
# - Include needed custom/core modules
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(CMakeSettings)
include(CTest)
include(CheckCXXSourceCompiles)

# - Core/C++/CUDA build and dependency settings
# For single-mode generators, default to Optimized with Debug if nothing is specified
if(NOT CMAKE_CONFIGURATION_TYPES)
  set(__DEFAULT_CMAKE_BUILD_TYPE RelWithDebInfo)
  if(CMAKE_BUILD_TYPE)
    set(__DEFAULT_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
  endif()
  set(CMAKE_BUILD_TYPE "${__DEFAULT_CMAKE_BUILD_TYPE}"
    CACHE STRING "Choose the type of build, options are: None Release MinSizeRel Debug RelWithDebInfo."
    FORCE
  )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

#----------------------------------------------------------------------------#
# User options
#----------------------------------------------------------------------------#
# Options
option(ADEPT_USE_SURF "Enable surface model navigation on GPU" OFF)
option(ADEPT_MIXED_PRECISION "Use B-field integration and surface model in mixed precision" OFF)
option(USE_SPLIT_KERNELS "Run split version of the transport kernels" OFF)
option(ADEPT_USE_EXT_BFIELD "Use external B field from file via the covfie library" OFF)
option(DEBUG_SINGLE_THREAD "Run transport kernels in single thread mode" OFF)
set(ADEPT_DEBUG_TRACK 0 CACHE STRING "Debug tracking level (0=off, >0=on with levels)")
option(ASYNC_MODE "Enable the async transport backend" OFF)
option(ENFORCE_STRICT_FLAGS "Enforce strict compiler flags, as also used in CMSSW" OFF)
option(ADEPT_ENABLE_POWER_METER "Compile the code for consumption measurement" OFF)


# Compile flag for AdePT SteppingActions on the GPU
set(ADEPT_STEPPINGACTION "NONE" CACHE STRING "SteppingAction: NONE|CMS|LHCb")
set_property(CACHE ADEPT_STEPPINGACTION PROPERTY STRINGS NONE CMS LHCb)

if(ADEPT_STEPPINGACTION STREQUAL "CMS")
  set(_ADEPT_STEPACTION_TYPE 1)
elseif(ADEPT_STEPPINGACTION STREQUAL "LHCb")
  set(_ADEPT_STEPACTION_TYPE 2)
else()
  set(_ADEPT_STEPACTION_TYPE 0)
endif()

message(STATUS "AdePT SteppingAction: ${Green}${ADEPT_STEPPINGACTION} ${ColorReset}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

  set(_warn_flags
    -Wall
    -Wextra
    -Wpedantic
    -Wno-non-template-friend
    -Wno-long-long
    -Wreturn-type
    -Wpessimizing-move
    -Wclass-memaccess
    -Wno-cast-function-type
    -Wno-unused-but-set-parameter
    -Wno-ignored-qualifiers
    -Wno-unused-parameter
    -Wunused
    -Wparentheses
    -Wnon-virtual-dtor
    -Wno-unused-local-typedefs
    -Wno-attributes
    -Wno-psabi
  )

  set(_error_flags
    -Werror=main
    -Werror=switch
    -Werror=pointer-arith
    -Werror=overlength-strings
    -Werror=overflow
    -Werror=array-bounds
    -Werror=format-contains-nul
    -Werror=type-limits
    -Werror=return-type
    -Werror=missing-braces
    -Werror=unused-value
    -Werror=unused-label
    -Werror=address
    -Werror=format
    -Werror=sign-compare
    -Werror=write-strings
    -Werror=delete-non-virtual-dtor
    -Werror=strict-aliasing
    -Werror=narrowing
    -Werror=unused-but-set-variable
    -Werror=reorder
    -Werror=unused-variable
    -Werror=conversion-null
    -Werror=return-local-addr
  )

  # In soft mode, convert all -Werror=foo to just -Wfoo
  if(NOT ENFORCE_STRICT_FLAGS)
    set(_converted_errors "")
    foreach(f IN LISTS _error_flags)
      string(REPLACE "-Werror=" "-W" _wflag "${f}")
      list(APPEND _converted_errors "${_wflag}")
    endforeach()
    set(_final_flags ${_warn_flags} ${_converted_errors})
  else()
    message(STATUS "Enabling strict compiler flags")
    set(_final_flags ${_warn_flags} ${_error_flags})
  endif()

  string(JOIN " " _final_flags_str ${_final_flags})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_final_flags_str}")
endif()


#----------------------------------------------------------------------------#
# Dependencies
#----------------------------------------------------------------------------#
# With CUDA language enabled above, this should find the toolkit alongside the compiler
find_package(CUDAToolkit REQUIRED)

#Find VecCore with correct backend
set(VecCore_VERSION 0.5.2)
set(VecCore_BACKEND CUDA)
find_package(VecCore ${VecCore_VERSION} REQUIRED COMPONENTS ${VecCore_BACKEND})
message(STATUS "Using VecCore version ${Cyan}${VecCore_VERSION}${ColorReset}")

# Before looking for other packages, try to find XercesC explicitly to avoid
# problems with G4HepEm not finding Geant4 11.1 even though we find it here.
find_package(XercesC REQUIRED)

# Find VecGeom geometry headers library
find_package(VecGeom REQUIRED)

### VecGeom Version check

# Use VecGeom_VERSION_STRING if defined, otherwise fall back to VecGeom_VERSION
if(NOT VecGeom_VERSION_STRING)
  set(VecGeom_VERSION_STRING ${VecGeom_VERSION})
endif()

# VecGeom might append build metadata (e.g. '+master.abc123') to the version string.
# Strip everything after the '+' to focus on the version core.
string(REGEX REPLACE "\\+.*$" "" CLEANED_VERSION "${VecGeom_VERSION_STRING}")


# Try to match the core version suffix to extract either:
# - an RC version like "rc3", "rc4"
# - a dev version like "dev.3", "dev.4"
# This works even if the full suffix is like "rc3.213" or "dev.4.99"
string(REGEX MATCH "2\\.0\\.0-(rc[0-9]+|dev\\.[0-9]+)" _match "${CLEANED_VERSION}")
set(MATCHED_SUFFIX "${CMAKE_MATCH_1}")

set(_vecgeom_dev_version 5)
set(_vecgeom_version_ok FALSE)

# Handle dev versions
if(MATCHED_SUFFIX MATCHES "dev\\.([0-9]+)")
  set(devver "${CMAKE_MATCH_1}")
  if(devver GREATER_EQUAL _vecgeom_dev_version)
    set(_vecgeom_version_ok TRUE)
  endif()
endif()

# Handle rc versions
if(MATCHED_SUFFIX MATCHES "rc\\.([0-9]+)")
  set(rcver "${CMAKE_MATCH_1}")
  if(rcver GREATER_EQUAL _vecgeom_dev_version)
    set(_vecgeom_version_ok TRUE)
  endif()
endif()

# Handle stable releases >= 2.0.0
if(CLEANED_VERSION VERSION_GREATER_EQUAL "2.0.0" AND NOT MATCHED_SUFFIX)
  set(_vecgeom_version_ok TRUE)
endif()

if(NOT _vecgeom_version_ok)
  message(FATAL_ERROR "AdePT requires VecGeom >= 2.0.0-dev.${_vecgeom_dev_version} or >= 2.0.0-rc.${_vecgeom_dev_version}, but found '${VecGeom_VERSION_STRING}'")
else()
  message(STATUS "Using VecGeom version ${Cyan}${VecGeom_VERSION_STRING}${ColorReset}")
endif()

### end version check for VecGeom

# Make sure VecGeom::vgdml is enabled
if(NOT TARGET VecGeom::vgdml)
  message(FATAL_ERROR "AdePT requires VecGeom compiled with GDML support")
endif()
if(ASYNC_MODE)
  add_compile_definitions(ASYNC_MODE)
  message(STATUS "${Green}Using the asynchronous transport backend${ColorReset}")
else()
  message(STATUS "${Red}Async backend is disabled${ColorReset}")
endif()
# Run split kernels
if (USE_SPLIT_KERNELS)
  if(ASYNC_MODE)
    add_compile_definitions(USE_SPLIT_KERNELS)
    message(STATUS "${Green}AdePT will run with split kernels${ColorReset}")
  else()
    message(FATAL_ERROR "In order to run with split kernels, AdePT must be compiled in ASYNC_MODE")
  endif()
endif()
# Debugging in single-thread mode
if (DEBUG_SINGLE_THREAD)
  add_compile_definitions("$<$<CONFIG:Debug>:DEBUG_SINGLE_THREAD>")
  message(STATUS "${Magenta}Transport is running in single-thread mode${ColorReset}")
endif()
# Check for RK integration precision
if (ADEPT_MIXED_PRECISION)
  message(STATUS "${Green}Using single precision in RK integration${ColorReset}")
  add_compile_definitions(ADEPT_MIXED_PRECISION)
else()
  message(STATUS "${Green}Using double precision in RK integration${ColorReset}")
endif()
# Check for surface model support in VecGeom
if(ADEPT_USE_SURF)
  if(VecGeom_SURF_FOUND)
    if (ADEPT_MIXED_PRECISION)
      message(STATUS "${Green}Using the surface model in VecGeom in mixed precision${ColorReset}")
    else()
      message(STATUS "${Green}Using the surface model in VecGeom in double precision${ColorReset}")
    endif()
    add_compile_definitions(ADEPT_USE_SURF)
  else()
    message(STATUS "${Magenta}No VecGeom surface support. Forcing ADEPT_USE_SURF to OFF${ColorReset}")
    set(ADEPT_USE_SURF OFF CACHE BOOL "Disable using the surface model" FORCE)
  endif()
else()
  message(STATUS "${Green}Using the solid model in VecGeom${ColorReset}")
endif()

# Find Geant4, optional for now
find_package(Geant4 QUIET)
if(Geant4_FOUND)
  message(STATUS "Using Geant4 version ${Cyan}${Geant4_VERSION}${ColorReset} from ${Geant4_INCLUDE_DIRS}")
  set(CMAKE_REQUIRED_INCLUDES ${Geant4_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${Geant4_LIBRARIES})
  # Check if Geant4 is supports G4VTrackingManager
  # Always available from v11.0, but experiments may backport it to earlier versions
  # so we do a compile check.
  check_cxx_source_compiles("
    #include \"G4VTrackingManager.hh\"
    #include \"G4Track.hh\"

    struct testtm_ : G4VTrackingManager {
      void HandOverOneTrack(G4Track*) override {}
    };

    int main() {
      testtm_ t;
      return 0;
    }" AdePT_HAS_G4VTRACKINGMANAGER)
else()
  message(STATUS "Did not find Geant4")
endif()

# Find HepMC3, used by integration examples to load realistic events
find_package(HepMC3 QUIET)
if(HepMC3_FOUND)
  message(STATUS "HepMC3 found ${HEPMC3_INCLUDE_DIR}")
  add_compile_definitions(HEPMC3_FOUND)
endif()

# Find G4VG, optional as requires Geant4
if(Geant4_FOUND)
  # Default to use FetchContent'd G4VG for current ease of use and until CI
  # includes an install. Also allows easy testing of upstream patches until
  # fully stable across projects.
  # NB: Though the option can be switched in a given build directory after the initial
  # build, if you have an install of G4VG that could/would be picked up, you may get
  # CMake-generate time warnings about safe runtime paths. Generally, you should
  # pick one way to pickup G4VG and stick with that throughout the lifetime of
  # the build directory.
  # The eventual fix for this will be to move to require an external G4VG only
  option(ADEPT_USE_BUILTIN_G4VG "Fetch and build G4VG as part of AdePT" ON)
  if(ADEPT_USE_BUILTIN_G4VG)
    # Fetch it locally
    include(FetchContent)
    FetchContent_Declare(
      g4vg
      GIT_REPOSITORY https://github.com/celeritas-project/g4vg
      GIT_TAG f4308d392a502b7bd7468938f8d9a63198d3d866 # v1.0.3
    )
    # G4VG builds static by default, so change this to shared to match current
    # way AdePT is built.
    # could also configure for PIC mode static.
    set(BUILD_SHARED_LIBS ON)
    FetchContent_MakeAvailable(g4vg)
    message(STATUS "Using FetchContent to build G4VG as part of AdePT")
  else()
    find_package(G4VG 1.0.3 REQUIRED)
    message(STATUS "Found G4VG: ${G4VG_DIR}")
  endif()
endif()

# Find covfie to load external fields from file. 
if(ADEPT_USE_EXT_BFIELD)
  # An 3D magnetic field map requires the covfie library.
  # The default is to use an installed covfie version. If no covfie installation is provided,
  # it is fetched from source and made available via FetchContent  
  
  message(STATUS "Compiling with external magnetic field support via covfie")
  add_compile_definitions(ADEPT_USE_EXT_BFIELD)

  find_package(covfie QUIET)
  if (covfie_FOUND)
    get_target_property(COVFIE_INCLUDE_DIR covfie::core INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "covfie found at: ${COVFIE_INCLUDE_DIR}")
  else()
    set(ADEPT_COVFIE_REPO "https://github.com/acts-project/covfie.git")
    set(ADEPT_COVFIE_BRANCH "v0.15.3") # latest version number at 06.08.2025

    message(STATUS "No covfie provided, fetching from: ${ADEPT_COVFIE_REPO} (branch: ${ADEPT_COVFIE_BRANCH})")

    include(FetchContent)
    FetchContent_Declare(
      covfie
      GIT_REPOSITORY ${ADEPT_COVFIE_REPO}
      GIT_TAG ${ADEPT_COVFIE_BRANCH}
    )

    set(COVFIE_PLATFORM_CUDA ON CACHE BOOL "Enable CUDA support in covfie" FORCE)
    FetchContent_MakeAvailable(covfie)

    FetchContent_GetProperties(covfie)
    if(NOT covfie_POPULATED)
      message(FATAL_ERROR "Fetching covfie failed!")
    endif()

  endif()
endif()

if(ADEPT_ENABLE_POWER_METER)
  find_package(Power_meter QUIET)
  if(Power_meter_FOUND)
    message(STATUS "Power_meter found ${Power_meter_INCLUDE_DIR}")
    add_compile_definitions(ENABLE_POWER_METER)
  else()
    # Fetch it locally
    include(FetchContent)

    FetchContent_Declare(
      power_meter
      GIT_REPOSITORY https://github.com/apt-sim/power_meter
      GIT_TAG 458fa2f4f72287d7128de25ed570d2803f0fbe89
    )

    FetchContent_MakeAvailable(power_meter)
    message(STATUS "Using FetchContent to build Power_meter")

    FetchContent_GetProperties(power_meter)
    if(NOT power_meter_POPULATED)
      message(FATAL_ERROR "Fetching Power_meter failed!")
    endif()

    add_compile_definitions(ENABLE_POWER_METER)
  endif()
endif()

# Set up debugging levels for CUDA:
# - For RelWithDebInfo (the default), generate line info to enable profiling.
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:--generate-line-info>")
# - For Debug, generate full debug information - this completely disables optimizations!
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--device-debug>")
# - For both, interleave the source in PTX to enhance the debugging experience.
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:Debug>>>:--source-in-ptx>")
# Disable warnings from the CUDA frontend about unknown GCC pragmas - let the compiler decide what it likes.
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe;--diag_suppress=unrecognized_gcc_pragma>")
# Add track debug as compile definition for CUDA if debugging requested
if(ADEPT_DEBUG_TRACK GREATER 0)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-DADEPT_DEBUG_TRACK=${ADEPT_DEBUG_TRACK}>")
endif()

find_package(G4HepEm CONFIG REQUIRED)
if(G4HepEm_FOUND)
  message(STATUS "G4HepEm found ${G4HepEm_INCLUDE_DIR}")
endif()

#----------------------------------------------------------------------------#
# Build Targets
#----------------------------------------------------------------------------#
set(ADEPT_G4_INTEGRATION_SRCS
  src/G4HepEmTrackingManagerSpecialized.cc
  src/AdePTTrackingManager.cc
  src/AdePTPhysics.cc
  src/HepEMPhysics.cc
  src/AdePTGeant4Integration.cpp
  src/AdePTConfigurationMessenger.cc
  src/AdePTTrackingManager.cu
)

add_library(CopCore INTERFACE)
target_include_directories(CopCore
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/AdePT/copcore/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)

cuda_rdc_add_library(AdePT_G4_integration SHARED
  ${ADEPT_G4_INTEGRATION_SRCS}
)
cuda_rdc_target_include_directories(AdePT_G4_integration
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)
# This needs to be after the cuda_rdc_target_include_directories when
# using v10 of CudaRdcUtils.cmake
cuda_rdc_add_library(AdePT::AdePT_G4_integration ALIAS AdePT_G4_integration)

cuda_rdc_target_link_libraries(AdePT_G4_integration
  PUBLIC
    CopCore
    VecGeom::vecgeom
    VecGeom::vgdml
    ${Geant4_LIBRARIES}
    G4HepEm::g4HepEm
    G4HepEm::g4HepEmData
    G4HepEm::g4HepEmInit
    G4HepEm::g4HepEmRun
    G4VG::g4vg
    CUDA::cudart
    $<$<BOOL:${ADEPT_ENABLE_POWER_METER}>:Power_meter::Power_meter>
)
if(ADEPT_USE_EXT_BFIELD)
  # link covfie if external magnetic field usage is enabled
  cuda_rdc_target_link_libraries(AdePT_G4_integration
    PUBLIC
      covfie::core
      covfie::cuda
  )
endif()

# must be defined when compiling AdePT_G4_integration_objects' source files into the objects lib
target_compile_definitions(AdePT_G4_integration_objects
  PRIVATE ADEPT_STEPACTION_TYPE=${_ADEPT_STEPACTION_TYPE}
)

# Export config macros to consumers (Athena, etc.)
target_compile_definitions(AdePT_G4_integration
  INTERFACE
    ADEPT_STEPACTION_TYPE=${_ADEPT_STEPACTION_TYPE}
    $<$<BOOL:${ASYNC_MODE}>:ASYNC_MODE>
    $<$<BOOL:${ADEPT_USE_SURF}>:ADEPT_USE_SURF>
    $<$<BOOL:${ADEPT_MIXED_PRECISION}>:ADEPT_MIXED_PRECISION>
    $<$<BOOL:${USE_SPLIT_KERNELS}>:USE_SPLIT_KERNELS>
    $<$<AND:$<BOOL:${DEBUG_SINGLE_THREAD}>,$<CONFIG:Debug>>:DEBUG_SINGLE_THREAD>
    $<$<BOOL:${ADEPT_USE_EXT_BFIELD}>:ADEPT_USE_EXT_BFIELD>
    $<$<BOOL:${ADEPT_ENABLE_POWER_METER}>:ADEPT_ENABLE_POWER_METER>
    $<$<BOOL:${ADEPT_DEBUG_TRACK}>:ADEPT_DEBUG_TRACK=${ADEPT_DEBUG_TRACK}>
)

#set_target_properties(AdePT_G4_integration
#  PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON
#    CUDA_RESOLVE_DEVICE_SYMBOLS ON
#)

# Optional library to activate NVTX annotations for profiling:
option(NVTX OFF "Add NVTX instrumentation for profiling (only for annotated examples)")
add_library(NVTX INTERFACE)
if(NVTX)
  target_link_libraries(NVTX INTERFACE nvToolsExt)
  target_compile_definitions(NVTX INTERFACE -DUSE_NVTX)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
  # This might become a separate option, e.g. "ADEPT_BUILD_EXAMPLES"
  add_subdirectory(examples)
endif()

####################################################################################################

include(CMakePackageConfigHelpers)
#Generate the configuration file from the template and save it to the build directory
configure_package_config_file(cmake/${PROJECT_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

#Install the libraries
cuda_rdc_install(TARGETS CopCore AdePT_G4_integration
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

#Install the headers
install(DIRECTORY include/AdePT
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

#Install the configuration file
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/AdePTConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

#Export the targets file
export(TARGETS CopCore AdePT_G4_integration
  FILE "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Targets.cmake"
)

#Install the targets file
install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
