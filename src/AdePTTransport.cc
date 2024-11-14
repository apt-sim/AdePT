// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTTransport.h>
#include <AdePT/integration/AdePTGeant4Integration.hh>
#include <AdePT/core/AdePTConfiguration.hh>

/// @brief Create one instance of the synchronous AdePTTransport implementation per call.
/// @param nThread Number of threads (to compute memory usage per thread).
/// @param nTrackSlot Number of track slots across all AdePT instances.
/// @param nHitSlot
/// @param verbosity
/// @param GPURegionNames
/// @param trackInAllRegions
/// @param cudaStackSize If != 0, change the cuda stack size.
/// @return
std::shared_ptr<AdePTTransportInterface> AdePTTransportFactory(unsigned int nThread, unsigned int nTrackSlot,
                                                               unsigned int nHitSlot, int verbosity,
                                                               std::vector<std::string> const *GPURegionNames,
                                                               bool trackInAllRegions, int cudaStackSize)
{
  auto adept = std::make_shared<AdePTTransport<AdePTGeant4Integration>>();

  // AdePT needs to be initialized here, since we know all needed Geant4 initializations are already finished
  adept->SetDebugLevel(verbosity);
  adept->SetTrackInAllRegions(trackInAllRegions);
  adept->SetGPURegionNames(GPURegionNames);
  if (cudaStackSize != 0) adept->SetCUDAStackLimit(cudaStackSize);

  const auto track_capacity = nTrackSlot / nThread;
  G4cout << "AdePT Allocated track capacity: " << track_capacity << " tracks" << G4endl;
  adept->SetTrackCapacity(track_capacity);

  const auto hit_buffer_capacity = nHitSlot / nThread;
  G4cout << "AdePT Allocated hit buffer capacity: " << hit_buffer_capacity << " slots" << G4endl;
  adept->SetHitBufferCapacity(hit_buffer_capacity);

  return adept;
}
