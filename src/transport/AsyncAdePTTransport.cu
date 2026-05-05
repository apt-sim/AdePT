// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/transport/AsyncAdePTTransport.cuh>

namespace AsyncAdePT {

void GPUstateDeleter::operator()(GPUstate *ptr)
{
  delete ptr;
}

} // namespace AsyncAdePT
