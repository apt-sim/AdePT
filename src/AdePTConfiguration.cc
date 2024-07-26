// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTConfiguration.hh>

#include <AdePT/core/AdePTTransport.h>
#include <AdePT/integration/AdePTGeant4Integration.hh>

namespace {
std::shared_ptr<AdePTTransportInterface> DefaultAdePTFactory(unsigned int nThread, unsigned int nTrackSlot,
                                                             unsigned int nHitSlot, int verbosity,
                                                             std::vector<std::string> const *GPURegionNames,
                                                             bool trackInAllRegions)
{
  auto adept = std::make_shared<AdePTTransport<AdePTGeant4Integration>>();

  // AdePT needs to be initialized here, since we know all needed Geant4 initializations are already finished
  adept->SetDebugLevel(verbosity);
  adept->SetTrackInAllRegions(trackInAllRegions);
  adept->SetGPURegionNames(GPURegionNames);

  const auto track_capacity = nTrackSlot / nThread;
  G4cout << "AdePT Allocated track capacity: " << track_capacity << " tracks" << G4endl;
  adept->SetTrackCapacity(track_capacity);

  const auto hit_buffer_capacity = nHitSlot / nThread;
  G4cout << "AdePT Allocated hit buffer capacity: " << hit_buffer_capacity << " slots" << G4endl;
  adept->SetHitBufferCapacity(hit_buffer_capacity);

  return adept;
}
} // namespace

AdePTConfiguration::AdePTConfiguration() : fAdePTConfigurationMessenger{new AdePTConfigurationMessenger(this)}
{
  if (!fAdePTFactoryFunction) fAdePTFactoryFunction = DefaultAdePTFactory;
}

AdePTConfiguration::~AdePTConfiguration() {}

std::shared_ptr<AdePTTransportInterface> AdePTConfiguration::CreateAdePTInstance(unsigned int nThread)
{
  if (fNThread == -1) {
    // The number of threads is only correct when the MTRunManager initialises AdePT.
    // Once the workers start up, they claim that the number of threads is 1.
    fNThread = nThread;
  }

  auto adept =
      fAdePTFactoryFunction(fNThread, 1024 * 1024 * GetMillionsOfTrackSlots(), 1024 * 1024 * GetMillionsOfHitSlots(),
                            GetVerbosity(), GetGPURegionNames(), GetTrackInAllRegions());

  // AdePT needs to be initialized here, since we know all needed Geant4 initializations are already finished
  adept->SetBufferThreshold(GetTransportBufferThreshold());
  adept->SetMaxBatch(2 * GetTransportBufferThreshold());

  return adept;
}
