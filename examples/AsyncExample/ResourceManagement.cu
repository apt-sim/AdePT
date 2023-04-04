// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#include "ResourceManagement.h"

#include "CopCore/Global.h"

namespace adeptint {

void cudaDeleter(void *ptr)
{
  COPCORE_CUDA_CHECK(cudaFree(ptr));
}

void cudaHostDeleter(void *ptr)
{
  COPCORE_CUDA_CHECK(cudaFreeHost(ptr));
}

void cudaStreamDeleter(cudaStream_t *stream)
{
  COPCORE_CUDA_CHECK(cudaStreamDestroy(*stream));
}

void cudaEventDeleter(cudaEvent_t *event)
{
  COPCORE_CUDA_CHECK(cudaEventDestroy(*event));
}
using unique_ptr_cudaEvent = std::unique_ptr<cudaEvent_t, void (*)(cudaEvent_t *)>;

} // namespace adeptint