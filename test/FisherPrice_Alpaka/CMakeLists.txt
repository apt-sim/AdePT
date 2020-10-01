# This makes sure cmake knows where to find alpaka
set(CMAKE_PREFIX_PATH "/install")
# Then we must specify that we want to use alpaka
find_package(alpaka REQUIRED)
alpaka_add_executable(FisherPriceAlpaka alpaka_fisher_price.cu)
target_link_libraries(FisherPriceAlpaka PUBLIC alpaka::alpaka)

