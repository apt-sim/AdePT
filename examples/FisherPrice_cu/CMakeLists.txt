# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Noddy example 1 of particle processing with GPU
add_executable(cufisher_price cufisher_price.cu)
target_link_libraries(cufisher_price PRIVATE AdePT CUDA::curand)

# Noddy example 2 of particle processing with GPU
add_executable(cufisher_price_v2 cufisher_price_v2.cu)
target_link_libraries(cufisher_price_v2 PRIVATE AdePT CUDA::curand)

