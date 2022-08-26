# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Our example depends on types in alpaka's CUDA backend, so we must have it enabled
set(alpaka_ACC_GPU_CUDA_ENABLE ON CACHE BOOL "Enable the CUDA GPU back-end" FORCE)

#We must specify that we want to use alpaka
find_package(alpaka 0.9.0 QUIET)

if(NOT alpaka_FOUND)
  message(STATUS "No alpaka install found, disabling build of FisherPrice_Alpaka")
  return()
else()
  alpaka_add_executable(FisherPriceAlpaka alpaka_fisher_price.cu)
  target_link_libraries(FisherPriceAlpaka PUBLIC alpaka::alpaka)

  alpaka_add_executable(unitTests_FisherPriceAlpaka unitTests_alpaka_fisher_price.cu)
  target_link_libraries(unitTests_FisherPriceAlpaka PUBLIC alpaka::alpaka)
endif()

