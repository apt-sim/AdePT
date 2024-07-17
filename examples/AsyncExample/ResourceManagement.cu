// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
#include "ResourceManagement.h"

#include "AdePT/copcore/Global.h"

namespace AsyncAdePT {

void cudaDeleter(void *ptr)
{
  COPCORE_CUDA_CHECK(cudaFree(ptr));
}

void cudaHostDeleter(void *ptr)
{
  COPCORE_CUDA_CHECK(cudaFreeHost(ptr));
}

void cudaStreamDeleter(void *stream)
{
  COPCORE_CUDA_CHECK(cudaStreamDestroy(*static_cast<cudaStream_t *>(stream)));
}

void cudaEventDeleter(void *event)
{
  COPCORE_CUDA_CHECK(cudaEventDestroy(*static_cast<cudaEvent_t *>(event)));
}

} // namespace AsyncAdePT
