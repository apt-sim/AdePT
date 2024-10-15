// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/AdePTConfiguration.hh>

#include <AdePT/integration/AdePTGeant4Integration.hh>

/// @brief Factory function to create AdePT instances.
/// Every AdePT transport implementation needs to provide this function to create
/// instances of the transport implementation. These might either be one instance
/// per thread, or share one instance across many threads. This is up to the
/// transport implementation.
extern std::shared_ptr<AdePTTransportInterface> AdePTTransportFactory(unsigned int nThread, unsigned int nTrackSlot,
                                                                      unsigned int nHitSlot, int verbosity,
                                                                      std::vector<std::string> const *GPURegionNames,
                                                                      bool trackInAllRegions);

AdePTConfiguration::AdePTConfiguration() : fAdePTConfigurationMessenger{new AdePTConfigurationMessenger(this)} {}

AdePTConfiguration::~AdePTConfiguration() {}

std::shared_ptr<AdePTTransportInterface> AdePTConfiguration::CreateAdePTInstance(unsigned int nThread)
{
  if (fNThread == -1) {
    // The number of threads is only correct when the MTRunManager initialises AdePT.
    // Once the workers start up, they claim that the number of threads is 1.
    fNThread = nThread;
  }

  auto adept =
      AdePTTransportFactory(fNThread, 1024 * 1024 * GetMillionsOfTrackSlots(), 1024 * 1024 * GetMillionsOfHitSlots(),
                            GetVerbosity(), GetGPURegionNames(), GetTrackInAllRegions());

  // AdePT needs to be initialized here, since we know all needed Geant4 initializations are already finished
  adept->SetBufferThreshold(GetTransportBufferThreshold());
  adept->SetMaxBatch(2 * GetTransportBufferThreshold());
  adept->SetCUDAStackLimit(GetCUDAStackLimit());

  return adept;
}
