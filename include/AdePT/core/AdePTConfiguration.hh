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
  void SetCallUserTrackingAction(bool callUserTrackingAction) { fCallUserTrackingAction = callUserTrackingAction; }
  void AddGPURegionName(std::string name) { fGPURegionNames.push_back(name); }
  void RemoveGPURegionName(std::string name) { fCPURegionNames.push_back(name); }
  void AddWDTRegionName(std::string name) { fWDTRegionNames.push_back(name); }
  void SetAdePTActivation(bool activateAdePT) { fAdePTActivated = activateAdePT; }
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; };
  void SetTransportBufferThreshold(int threshold) { fTransportBufferThreshold = threshold; }
  void SetMillionsOfTrackSlots(double millionSlots) { fMillionsOfTrackSlots = millionSlots; }
  void SetMillionsOfLeakSlots(double millionSlots) { fMillionsOfLeakSlots = millionSlots; }
  void SetMillionsOfHitSlots(double millionSlots) { fMillionsOfHitSlots = millionSlots; }
  void SetHitBufferFlushThreshold(float threshold) { fHitBufferFlushThreshold = threshold; }
  void SetCPUCapacityFactor(float CPUCapacityFactor) { fCPUCapacityFactor = CPUCapacityFactor; }
  void SetHitBufferSafetyFactor(double HitBufferSafetyFactor) { fHitBufferSafetyFactor = HitBufferSafetyFactor; }
  void SetAdePTSeed(int seed) { fAdePTSeed = static_cast<uint64_t>(seed); }

  void SetCUDAStackLimit(int limit) { fCUDAStackLimit = limit; }
  void SetCUDAHeapLimit(int limit) { fCUDAHeapLimit = limit; }
  void SetLastNParticlesOnCPU(int Nparticles) { fLastNParticlesOnCPU = Nparticles; }
  void SetMaxWDTIter(int maxIter) { fMaxWDTIter = maxIter; }
  void SetWDTKineticEnergyLimit(double ekin) { fWDTKineticEnergyLimit = ekin; }
  void SetSpeedOfLight(bool speedOfLight) { fSpeedOfLight = speedOfLight; }
  void SetMultipleStepsInMSCWithTransportation(bool setMultipleSteps)
  {
    fSetMultipleStepsInMSCWithTransportation = setMultipleSteps;
  }
  void SetEnergyLossFluctuation(bool setELossFluct) { fSetEnergyLossFluctuation = setELossFluct; }

  // We temporarily load VecGeom geometry from GDML
  void SetVecGeomGDML(std::string filename) { fVecGeomGDML = filename; }

  // loading external magnetic field from .cf file
  void SetCovfieBfieldFile(std::string filename) { fCovfieBfieldFile = filename; }

  bool GetTrackInAllRegions() { return fTrackInAllRegions; }
  bool GetCallUserSteppingAction() { return fCallUserSteppingAction; }
  bool GetCallUserTrackingAction() { return fCallUserTrackingAction; }
  bool GetSpeedOfLight() { return fSpeedOfLight; }
  bool GetMultipleStepsInMSCWithTransportation() { return fSetMultipleStepsInMSCWithTransportation; }
  bool GetEnergyLossFluctuation() { return fSetEnergyLossFluctuation; }
  bool IsAdePTActivated() { return fAdePTActivated; }
  int GetNumThreads() { return fNumThreads; };
  int GetVerbosity() { return fVerbosity; };
  int GetTransportBufferThreshold() { return fTransportBufferThreshold; }
  int GetCUDAStackLimit() { return fCUDAStackLimit; }
  int GetCUDAHeapLimit() { return fCUDAHeapLimit; }
  uint64_t GetAdePTSeed() { return fAdePTSeed; }

  unsigned short GetLastNParticlesOnCPU() { return fLastNParticlesOnCPU; }
  unsigned short GetMaxWDTIter() { return fMaxWDTIter; }
  double GetWDTKineticEnergyLimit() { return fWDTKineticEnergyLimit; }
  float GetHitBufferFlushThreshold() { return fHitBufferFlushThreshold; }
  float GetCPUCapacityFactor() { return fCPUCapacityFactor; }
  double GetHitBufferSafetyFactor() { return fHitBufferSafetyFactor; }
  double GetMillionsOfTrackSlots() { return fMillionsOfTrackSlots; }
  double GetMillionsOfLeakSlots() { return fMillionsOfLeakSlots; }
  double GetMillionsOfHitSlots() { return fMillionsOfHitSlots; }
  std::vector<std::string> *GetGPURegionNames() { return &fGPURegionNames; }
  std::vector<std::string> *GetCPURegionNames() { return &fCPURegionNames; }
  const std::vector<std::string> &GetWDTRegionNames() const { return fWDTRegionNames; }

  // Temporary
  std::string GetVecGeomGDML() { return fVecGeomGDML; }
  std::string GetCovfieBfieldFile() { return fCovfieBfieldFile; } // todo add #ifdef guards?

private:
  bool fTrackInAllRegions{false};
  bool fCallUserSteppingAction{false};
  bool fCallUserTrackingAction{false};
  bool fSpeedOfLight{false};
  bool fSetMultipleStepsInMSCWithTransportation{false};
  bool fSetEnergyLossFluctuation{false};
  bool fAdePTActivated{true};
  int fNumThreads{-1};
  int fVerbosity{0};
  int fTransportBufferThreshold{200};
  int fCUDAStackLimit{0};
  int fCUDAHeapLimit{0};
  uint64_t fAdePTSeed{1234567};
  float fHitBufferFlushThreshold{0.8};
  float fCPUCapacityFactor{2.5};
  double fHitBufferSafetyFactor{1.5};
  double fMillionsOfTrackSlots{1};
  double fMillionsOfLeakSlots{1};
  double fMillionsOfHitSlots{1};
  unsigned short fLastNParticlesOnCPU{0};
  unsigned short fMaxWDTIter{5};
  double fWDTKineticEnergyLimit{0.2}; // 200 keV

  std::vector<std::string> fGPURegionNames{};
  std::vector<std::string> fCPURegionNames{};
  std::vector<std::string> fWDTRegionNames{};

  std::string fVecGeomGDML{""};
  std::string fCovfieBfieldFile{""};
  std::unique_ptr<AdePTConfigurationMessenger> fAdePTConfigurationMessenger;
};

#endif
