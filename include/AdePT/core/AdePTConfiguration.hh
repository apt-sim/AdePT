// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_CONFIGURATION_HH
#define ADEPT_CONFIGURATION_HH

#include <AdePT/integration/AdePTConfigurationMessenger.hh>
#include <AdePT/core/AdePTTransportInterface.hh>

#include <memory>
#include <string>
#include <vector>

/// @brief Factory function to create AdePT instances.
/// Every AdePT transport implementation needs to provide this function to create
/// instances of the transport implementation. These might either be one instance
/// per thread, or share one instance across many threads. This is up to the
/// transport implementation.
std::shared_ptr<AdePTTransportInterface> AdePTTransportFactory(unsigned int nThread, unsigned int nTrackSlot,
                                                               unsigned int nHitSlot, int verbosity,
                                                               std::vector<std::string> const *GPURegionNames,
                                                               bool trackInAllRegions, int cudaStackSize);

/// @brief Create and configure instances of an AdePT transport implementation.
///
class AdePTConfiguration {
public:
  AdePTConfiguration() : fAdePTConfigurationMessenger{new AdePTConfigurationMessenger(this)} {}
  ~AdePTConfiguration() {}
  void SetNumThreads(int numThreads) { fNumThreads = numThreads; }
  void SetTrackInAllRegions(bool trackInAllRegions) { fTrackInAllRegions = trackInAllRegions; }
  void SetCallUserSteppingAction(bool callUserSteppingAction) { fCallUserSteppingAction = callUserSteppingAction; }
  void SetCallPostUserTrackingAction(bool callPostUserTrackingAction)
  {
    fCallPostUserTrackingAction = callPostUserTrackingAction;
  }
  void AddGPURegionName(std::string name) { fGPURegionNames.push_back(name); }
  void SetAdePTActivation(bool activateAdePT) { fAdePTActivated = activateAdePT; }
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; };
  void SetTransportBufferThreshold(int threshold) { fTransportBufferThreshold = threshold; }
  void SetMillionsOfTrackSlots(double millionSlots) { fMillionsOfTrackSlots = millionSlots; }
  void SetMillionsOfHitSlots(double millionSlots) { fMillionsOfHitSlots = millionSlots; }
  void SetHitBufferFlushThreshold(float threshold) { fHitBufferFlushThreshold = threshold; }
  void SetCPUCapacityFactor(float CPUCapacityFactor) { fCPUCapacityFactor = CPUCapacityFactor; }

  void SetCUDAStackLimit(int limit) { fCUDAStackLimit = limit; }
  void SetCUDAHeapLimit(int limit) { fCUDAHeapLimit = limit; }
  void SetLastNParticlesOnCPU(int Nparticles) { fLastNParticlesOnCPU = Nparticles; }
  void SetSpeedOfLightCmd(bool speedOfLight) { fSpeedOfLight = speedOfLight; }

  // We temporarily load VecGeom geometry from GDML
  void SetVecGeomGDML(std::string filename) { fVecGeomGDML = filename; }

  // loading external magnetic field from .cf file
  void SetCovfieBfieldFile(std::string filename) { fCovfieBfieldFile = filename; }

  bool GetTrackInAllRegions() { return fTrackInAllRegions; }
  bool GetCallUserSteppingAction() { return fCallUserSteppingAction; }
  bool GetCallPostUserTrackingAction() { return fCallPostUserTrackingAction; }
  bool GetSpeedOfLight() { return fSpeedOfLight; }
  bool IsAdePTActivated() { return fAdePTActivated; }
  int GetNumThreads() { return fNumThreads; };
  int GetVerbosity() { return fVerbosity; };
  int GetTransportBufferThreshold() { return fTransportBufferThreshold; }
  int GetCUDAStackLimit() { return fCUDAStackLimit; }
  int GetCUDAHeapLimit() { return fCUDAHeapLimit; }
  unsigned short GetLastNParticlesOnCPU() { return fLastNParticlesOnCPU; }
  float GetHitBufferFlushThreshold() { return fHitBufferFlushThreshold; }
  float GetCPUCapacityFactor() { return fCPUCapacityFactor; }
  double GetMillionsOfTrackSlots() { return fMillionsOfTrackSlots; }
  double GetMillionsOfHitSlots() { return fMillionsOfHitSlots; }
  std::vector<std::string> *GetGPURegionNames() { return &fGPURegionNames; }

  // Temporary
  std::string GetVecGeomGDML() { return fVecGeomGDML; }
  std::string GetCovfieBfieldFile() { return fCovfieBfieldFile; } // todo add #ifdef guards?

private:
  bool fTrackInAllRegions{false};
  bool fCallUserSteppingAction{false};
  bool fCallPostUserTrackingAction{false};
  bool fSpeedOfLight{false};
  bool fAdePTActivated{true};
  int fNumThreads;
  int fVerbosity{0};
  int fTransportBufferThreshold{200};
  int fCUDAStackLimit{0};
  int fCUDAHeapLimit{0};
  float fHitBufferFlushThreshold{0.8};
  float fCPUCapacityFactor{2.5};
  double fMillionsOfTrackSlots{1};
  double fMillionsOfHitSlots{1};
  unsigned short fLastNParticlesOnCPU{0};

  std::vector<std::string> fGPURegionNames{};
  int fNThread = -1;

  std::string fVecGeomGDML{""};
  std::string fCovfieBfieldFile{""};
  std::unique_ptr<AdePTConfigurationMessenger> fAdePTConfigurationMessenger;
};

#endif
