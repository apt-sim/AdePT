#We must specify that we want to use alpaka
find_package(alpaka)

if(NOT alpaka_FOUND)
  message(STATUS "No alpaka install found, disabling build of FisherPrice_Alpaka")
  return()
else()  
  alpaka_add_executable(FisherPriceAlpaka alpaka_fisher_price.cu)
  target_link_libraries(FisherPriceAlpaka PUBLIC alpaka::alpaka)
endif()

