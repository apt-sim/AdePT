# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

#We must specify that we want to use alpaka
find_package(alpaka QUIET)

if(NOT alpaka_FOUND)
  message(STATUS "No alpaka install found, disabling build of FisherPrice_Alpaka")
  return()
else()
  alpaka_add_executable(FisherPriceAlpaka alpaka_fisher_price.cu)
  target_link_libraries(FisherPriceAlpaka PUBLIC alpaka::alpaka)

  alpaka_add_executable(unitTests_FisherPriceAlpaka unitTests_alpaka_fisher_price.cu)
  target_link_libraries(unitTests_FisherPriceAlpaka PUBLIC alpaka::alpaka)
endif()

