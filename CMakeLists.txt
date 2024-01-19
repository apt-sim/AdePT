# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)

# Record the command line invoking the cmake command. Replay with recmake_initial.sh.
include(cmake/RecordCmdLine.cmake)

project(Adept
  VERSION 0.1.0
  DESCRIPTION "Accelerated demonstrator of electromagnetic Particle Transport"
  LANGUAGES C CXX CUDA)

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
    CACHE STRING "Choose the type of build, options are: None Release MinSizeRel Debug RelWithDebInfo MinSizeRel."
    FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

# (for now internal) CopCore dependency
add_subdirectory(base/inc/CopCore)

# With CUDA language enabled above, this should find the toolkit alongside the compiler
find_package(CUDAToolkit REQUIRED)

#Find VecCore with correct backend
set(VecCore_VERSION 0.5.2)
set(VecCore_BACKEND CUDA)
find_package(VecCore ${VecCore_VERSION} REQUIRED COMPONENTS ${VecCore_BACKEND})
message(STATUS "Using VecCore version ${VecCore_VERSION}")

option(ADEPT_USE_SURF "Enable surface model navigation on GPU" OFF)

# Before looking for other packages, try to find XercesC explicitly to avoid
# problems with G4HepEm not finding Geant4 11.1 even though we find it here.
find_package(XercesC REQUIRED)

# Find VecGeom geometry headers library
set(VecGeom_VERSION_REQ 2.0.0-dev.3)
find_package(VecGeom REQUIRED)
# Older versions did not provide VecGeom_VERSION_STRING, but only VecGeom_VERSION
if (NOT VecGeom_VERSION_STRING)
  set(VecGeom_VERSION_STRING ${VecGeom_VERSION})
endif()
if (VecGeom_VERSION_STRING STRLESS VecGeom_VERSION_REQ)
  message(FATAL_ERROR "AdePT requires at least VecGeom ${VecGeom_VERSION_REQ}, found ${VecGeom_VERSION_STRING}" )
else()
  message(STATUS "Using VecGeom version ${VecGeom_VERSION_STRING}")
endif()
# make sure we import VecGeom architecture flags - is this needed?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECGEOM_CXX_FLAGS}")
# Make sure VecGeom::vgdml is enabled
if(NOT TARGET VecGeom::vgdml)
  message(FATAL_ERROR "AdePT requires VecGeom compiled with GDML support")
endif()
# Check for surface model support in VecGeom
if (ADEPT_USE_SURF)
  if (VecGeom_SURF_FOUND)
    message(STATUS "Surface model in VecGeom is enabled")
    add_compile_definitions(ADEPT_USE_SURF)
  else()
    message(STATUS "Forcing ADEPT_USE_SURF to OFF (surface support not found in VecGeom)")
    set(ADEPT_USE_SURF OFF CACHE BOOL "Disable using the surface model" FORCE)
  endif()
endif()

# Find Geant4, optional for now
find_package(Geant4 QUIET)
if(Geant4_FOUND)
  message(STATUS "Using Geant4 version ${Geant4_VERSION} from ${Geant4_INCLUDE_DIRS}")
  set(CMAKE_REQUIRED_INCLUDES ${Geant4_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${Geant4_LIBRARIES})
  check_cxx_source_compiles("
  #include \"G4VFastSimulationModel.hh\"
  class testflush_ : public G4VFastSimulationModel {
    G4bool IsApplicable(const G4ParticleDefinition&) {return true;}
    G4bool ModelTrigger(const G4FastTrack &) {return true;}
    void DoIt(const G4FastTrack&, G4FastStep&) {}
    void Flush() final override {}
    public: testflush_() : G4VFastSimulationModel(\"\") {}
  };
  int main() { testflush_ model; return 0; }" Geant4_flush_FOUND)
else()
  message(STATUS "Did not find Geant4")
endif()

# Find HepMC3, used by integration examples to load realistic events
find_package(HepMC3 QUIET)

if(HepMC3_FOUND)
  message(STATUS "HepMC3 found ${HEPMC3_INCLUDE_DIR}")
  add_compile_definitions(HEPMC3_FOUND)
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

# Add external dependencies before our own code to allow checking for the
# targets and depend on them.
add_subdirectory(external)

if(BUILD_TESTING)
  set(TESTING_GDML "${PROJECT_BINARY_DIR}/trackML.gdml")
  set(CMS2018_GDML "${PROJECT_BINARY_DIR}/cms2018.gdml")
  file(DOWNLOAD https://gitlab.cern.ch/VecGeom/VecGeom/raw/v1.2.0/persistency/gdml/gdmls/trackML.gdml "${TESTING_GDML}"
    EXPECTED_HASH SHA256=2c53e0f2f4673c61f8679702532647bf71ec64c1613aae330fa835e7d7087064)
  file(DOWNLOAD https://gitlab.cern.ch/VecGeom/VecGeom/raw/v1.2.0/persistency/gdml/gdmls/cms2018.gdml "${CMS2018_GDML}"
    EXPECTED_HASH SHA256=a6538d8f196fbfe4c14e806df3439d5a0d7050d538d364faabe882c750970e63)
endif()

# Builds...
# Target for use of AdePT
# NB: NOT complete yet due to dependence of AdePT/LoopNavigator.h on VecGeom
#     This would be another INTERFACE link library, but needs care because it
#     involves VecGeom+CUDA, so usage requirements need some care and thought
add_library(AdePT INTERFACE)
target_include_directories(AdePT
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/base/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/magneticfield/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/tracking/inc>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(AdePT INTERFACE CopCore::CopCore)

# Optional library to activate NVTX annotations for profiling:
set(NVTX OFF CACHE BOOL "Add NVTX instrumentation for profiling (only for annotated examples)")
add_library(NVTX INTERFACE)
if(NVTX)
  target_link_libraries(NVTX INTERFACE nvToolsExt)
  target_compile_definitions(NVTX INTERFACE -DUSE_NVTX)
endif()

if(TEST)
  add_definitions(-DTEST)
  message(STATUS "Testing enabled")
endif()

add_subdirectory(tracking)
add_subdirectory(test)
add_subdirectory(examples)
