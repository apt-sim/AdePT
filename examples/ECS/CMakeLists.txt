# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

add_executable(ecs_demo
  ecs.cu
)
set_property(TARGET ecs_demo PROPERTY CUDA_STANDARD 17)

target_link_libraries(ecs_demo CopCore)

target_compile_options(ecs_demo PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Wall;>")
target_compile_options(ecs_demo PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>")
target_compile_options(ecs_demo PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>")