# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# - Project wide CMake settings
# - Though we check for some absolute paths, ensure there are no others
set(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)

# - Never export to or search in user/system package registry
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON)

# - Force project directories to appear first in any list of includes
set(CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE ON)

# - Only relink shared libs when interface changes
set(CMAKE_LINK_DEPENDS_NO_SHARED ON)

# - Only report newly installed files
set(CMAKE_INSTALL_MESSAGE LAZY)

#-------------------------------------------------------------------------------
# Output Directories for Build products
#-------------------------------------------------------------------------------
# In all cases, build products are stored in a directory tree rooted
# in a directory named ``BuildProducts`` under the ``PROJECT_BINARY_DIRECTORY``.
set(BASE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/BuildProducts")

# - Base subdirectories on GNU style, which also provides a common
include(GNUInstallDirs)

# - Single mode case
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BASE_OUTPUT_DIRECTORY}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${BASE_OUTPUT_DIRECTORY}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${BASE_OUTPUT_DIRECTORY}/${CMAKE_INSTALL_LIBDIR}")

# - Multimode case
foreach(_conftype ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${_conftype} _conftype_uppercase)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_conftype_uppercase}
    "${BASE_OUTPUT_DIRECTORY}/${_conftype}/${CMAKE_INSTALL_BINDIR}"
    )
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_conftype_uppercase}
    "${BASE_OUTPUT_DIRECTORY}/${_conftype}/${CMAKE_INSTALL_LIBDIR}"
    )
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_conftype_uppercase}
    "${BASE_OUTPUT_DIRECTORY}/${_conftype}/${CMAKE_INSTALL_LIBDIR}"
    )
endforeach()

