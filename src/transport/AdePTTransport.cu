// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/transport/AdePTTransport.cuh>

namespace adept::transport {

void GPUstateDeleter::operator()(GPUstate *ptr)
{
  delete ptr;
}

} // namespace adept::transport
