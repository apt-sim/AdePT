// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_ASYNC_MODE
#include <AdePT/core/AdePTTransport.cuh>
#include <AdePT/integration/AdePTGeant4Integration.hh>
#else
#include <AdePT/core/AsyncAdePTTransport.cuh>
#endif

#ifndef ADEPT_ASYNC_MODE

// Explicit instantiation of the ShowerGPU<AdePTGeant4Integration> function
namespace adept_impl {
template void ShowerGPU<AdePTGeant4Integration>(AdePTGeant4Integration &, int, adeptint::TrackBuffer &, GPUstate &,
                                                HostScoring *, HostScoring *, int, uint64_t);
} // namespace adept_impl

#endif
