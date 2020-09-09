# - Minimal test of cuda compilation
add_executable(example_add example_add.cu)

# Noddy example of particle processing on CPU
add_executable(fisher_price fisher_price.cpp)
# Noddy example of particle processing with GPU
add_executable(cufisher_price cufisher_price.cu)
target_link_libraries(cufisher_price PRIVATE CUDA::curand)
