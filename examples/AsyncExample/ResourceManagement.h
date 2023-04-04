// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#ifndef RESOURCE_MANAGEMENT_CUH
#define RESOURCE_MANAGEMENT_CUH

#include <memory>

namespace adeptint {
void cudaDeleter(void *ptr);
void cudaHostDeleter(void *ptr);

template <typename T = void>
using unique_ptr_cuda = std::unique_ptr<T, void (*)(void *)>;

void cudaStreamDeleter(cudaStream_t *stream);
using unique_ptr_cudaStream = std::unique_ptr<cudaStream_t, void (*)(cudaStream_t *)>;
void cudaEventDeleter(cudaEvent_t *event);
using unique_ptr_cudaEvent = std::unique_ptr<cudaEvent_t, void (*)(cudaEvent_t *)>;

} // namespace adeptint

#endif // RESOURCE_MANAGEMENT_CUH