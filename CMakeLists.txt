# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18...3.24)

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
include(CudaRdcUtils)

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

set(CMAKE_CXX_STANDARD 17)
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

#----------------------------------------------------------------------------#
# Dependencies
#----------------------------------------------------------------------------#
# With CUDA language enabled above, this should find the toolkit alongside the compiler
find_package(CUDAToolkit REQUIRED)

#Find VecCore with correct backend
set(VecCore_VERSION 0.5.2)
set(VecCore_BACKEND CUDA)
find_package(VecCore ${VecCore_VERSION} REQUIRED COMPONENTS ${VecCore_BACKEND})
message(STATUS "Using VecCore version ${VecCore_VERSION}")

# Before looking for other packages, try to find XercesC explicitly to avoid
# problems with G4HepEm not finding Geant4 11.1 even though we find it here.
find_package(XercesC REQUIRED)

# Find VecGeom geometry headers library
set(VecGeom_VERSION_REQ 2.0.0-dev.3)
find_package(VecGeom REQUIRED)
# Older versions did not provide VecGeom_VERSION_STRING, but only VecGeom_VERSION
if(NOT VecGeom_VERSION_STRING)
  set(VecGeom_VERSION_STRING ${VecGeom_VERSION})
endif()
if(VecGeom_VERSION_STRING STRLESS VecGeom_VERSION_REQ)
  message(FATAL_ERROR "AdePT requires at least VecGeom ${VecGeom_VERSION_REQ}, found ${VecGeom_VERSION_STRING}" )
else()
  message(STATUS "Using VecGeom version ${VecGeom_VERSION_STRING}")
endif()
# Make sure VecGeom::vgdml is enabled
if(NOT TARGET VecGeom::vgdml)
  message(FATAL_ERROR "AdePT requires VecGeom compiled with GDML support")
endif()
# Make sure VecGeom has CUDA support)
if(NOT TARGET VecGeom::vecgeomcuda)
  message(FATAL_ERROR "AdePT requires VecGeom compiled with CUDA support")
endif()
# Check for surface model support in VecGeom
if(ADEPT_USE_SURF)
  if(VecGeom_SURF_FOUND)
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

# G4VG - external for now
# Nominally we should build it as a static/object library to not expose it downstream.
# Left as installed package until co-working with Celeritas established in celer-adept in case of
# duplicate symbols.
if(NOT G4VG_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    g4vg
    URL https://github.com/celeritas-project/g4vg/archive/c1f4a9315ab2c23e5ba0043f7c3422586a738799.zip)
  FetchContent_MakeAvailable(g4vg)
endif()

# Because of the structure of RDC libraries, we need to ensure full linking on
# platforms that may have "--as-needed" as the default
add_link_options("$<HOST_LINK:$<$<CXX_COMPILER_ID:GNU,Clang>:LINKER:--no-as-needed>>")

# Set up debugging levels for CUDA:
# - For RelWithDebInfo (the default), generate line info to enable profiling.
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:--generate-line-info>")
# - For Debug, generate full debug information - this completely disables optimizations!
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--device-debug>")
# - For both, interleave the source in PTX to enhance the debugging experience.
add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:RelWithDebInfo>,$<CONFIG:Debug>>>:--source-in-ptx>")

# Disable warnings from the CUDA frontend about unknown GCC pragmas - let the compiler decide what it likes.
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe;--diag_suppress=unrecognized_gcc_pragma>")

find_package(G4HepEm CONFIG REQUIRED)
if(G4HepEm_FOUND)
  message(STATUS "G4HepEm found ${G4HepEm_INCLUDE_DIR}")
endif()

#----------------------------------------------------------------------------#
# Build Targets
#----------------------------------------------------------------------------#
set(ADEPT_G4_INTEGRATION_SRCS
  src/AdePTTrackingManager.cc
  src/AdePTTrackingManager.cu
  src/AdePTPhysics.cc
  src/HepEMPhysics.cc
  src/AdePTGeant4Integration.cpp
  src/AdePTConfigurationMessenger.cc
)

cuda_rdc_add_library(CopCore INTERFACE)
cuda_rdc_add_library(CopCore::CopCore ALIAS CopCore)
cuda_rdc_target_include_directories(CopCore
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/AdePT/copcore/>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)

cuda_rdc_add_library(AdePT_G4_integration SHARED ${ADEPT_G4_INTEGRATION_SRCS})
cuda_rdc_add_library(AdePT::AdePT_G4_integration ALIAS AdePT_G4_integration)
cuda_rdc_target_include_directories(AdePT_G4_integration
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)
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
)

set_target_properties(AdePT_G4_integration
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

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

#Install the configuration file and support modules needed downstream
install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/AdePTConfig.cmake"
  "${PROJECT_SOURCE_DIR}/cmake/CudaRdcUtils.cmake"
  "${PROJECT_SOURCE_DIR}/cmake/FindVecGeom.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

#Install the targets file
cuda_rdc_install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
