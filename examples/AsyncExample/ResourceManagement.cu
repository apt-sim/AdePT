// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#include "ResourceManagement.h"

#include "AdePT/copcore/Global.h"

namespace AsyncAdePT {

void freeCuda(void *ptr)
{
  if (ptr) COPCORE_CUDA_CHECK(cudaFree(ptr));
}

void freeCudaHost(void *ptr)
{
  if (ptr) COPCORE_CUDA_CHECK(cudaFreeHost(ptr));
}

void freeCudaStream(void *stream)
{
  if (stream) COPCORE_CUDA_CHECK(cudaStreamDestroy(*static_cast<cudaStream_t *>(stream)));
}

void freeCudaEvent(void *event)
{
  if (event) COPCORE_CUDA_CHECK(cudaEventDestroy(*static_cast<cudaEvent_t *>(event)));
}

} // namespace AsyncAdePT
