// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTTransport.cuh>
#include <AdePT/core/AsyncAdePTTransport.cuh>
#include <AdePT/integration/AdePTGeant4Integration.hh>

// Explicit instantiation of the ShowerGPU<AdePTGeant4Integration> function
namespace adept_impl {
template void ShowerGPU<AdePTGeant4Integration>(AdePTGeant4Integration &, int, adeptint::TrackBuffer &, GPUstate &,
                                                HostScoring *, HostScoring *);
} // namespace adept_impl

// namespace async_adept_impl
// {
// void AsyncAdePTTransport::TransportLoop();
// }
