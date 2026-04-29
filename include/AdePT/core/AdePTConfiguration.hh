// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_CONFIGURATION_HH
#define ADEPT_CONFIGURATION_HH

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class AdePTConfigurationMessenger;

/// @brief Create and configure instances of an AdePT transport implementation.
///
class AdePTConfiguration {
public:
  AdePTConfiguration();
  ~AdePTConfiguration();
  void SetNumThreads(int numThreads) { fNumThreads = numThreads; }
  void SetTrackInAllRegions(bool trackInAllRegions) { fTrackInAllRegions = trackInAllRegions; }
  void SetCallUserSteppingAction(bool callUserSteppingAction) { fCallUserSteppingAction = callUserSteppingAction; }
  void SetCallUserTrackingAction(bool callUserTrackingAction) { fCallUserTrackingAction = callUserTrackingAction; }
  void AddGPURegionName(std::string name) { fGPURegionNames.push_back(name); }
  void RemoveGPURegionName(std::string name) { fCPURegionNames.push_back(name); }
  void AddWDTRegionName(std::string name) { fWDTRegionNames.push_back(name); }
  void AddDeadRegionName(std::string name) { fDeadRegionNames.push_back(name); }
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; };
  void SetMillionsOfTrackSlots(double millionSlots) { fMillionsOfTrackSlots = millionSlots; }
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

  bool GetTrackInAllRegions() const { return fTrackInAllRegions; }
  bool GetCallUserSteppingAction() const { return fCallUserSteppingAction; }
  bool GetCallUserTrackingAction() const { return fCallUserTrackingAction; }
  bool GetSpeedOfLight() const { return fSpeedOfLight; }
  bool GetMultipleStepsInMSCWithTransportation() const { return fSetMultipleStepsInMSCWithTransportation; }
  bool GetEnergyLossFluctuation() const { return fSetEnergyLossFluctuation; }
  int GetNumThreads() const { return fNumThreads; };
  int GetVerbosity() const { return fVerbosity; };
  int GetCUDAStackLimit() const { return fCUDAStackLimit; }
  int GetCUDAHeapLimit() const { return fCUDAHeapLimit; }
  uint64_t GetAdePTSeed() const { return fAdePTSeed; }

  unsigned short GetLastNParticlesOnCPU() const { return fLastNParticlesOnCPU; }
  unsigned short GetMaxWDTIter() const { return fMaxWDTIter; }
  double GetWDTKineticEnergyLimit() const { return fWDTKineticEnergyLimit; }
  float GetHitBufferFlushThreshold() const { return fHitBufferFlushThreshold; }
  float GetCPUCapacityFactor() const { return fCPUCapacityFactor; }
  double GetHitBufferSafetyFactor() const { return fHitBufferSafetyFactor; }
  double GetMillionsOfTrackSlots() const { return fMillionsOfTrackSlots; }
  double GetMillionsOfHitSlots() const { return fMillionsOfHitSlots; }
  const std::vector<std::string> *GetGPURegionNames() const { return &fGPURegionNames; }
  const std::vector<std::string> *GetCPURegionNames() const { return &fCPURegionNames; }
  const std::vector<std::string> &GetWDTRegionNames() const { return fWDTRegionNames; }
  const std::vector<std::string> &GetDeadRegionNames() const { return fDeadRegionNames; }

  // Temporary
  std::string GetVecGeomGDML() const { return fVecGeomGDML; }
  std::string GetCovfieBfieldFile() const { return fCovfieBfieldFile; } // todo add #ifdef guards?

private:
  bool fTrackInAllRegions{false};
  bool fCallUserSteppingAction{false};
  bool fCallUserTrackingAction{false};
  bool fSpeedOfLight{false};
  bool fSetMultipleStepsInMSCWithTransportation{false};
  bool fSetEnergyLossFluctuation{false};
  int fNumThreads{-1};
  int fVerbosity{0};
  int fCUDAStackLimit{0};
  int fCUDAHeapLimit{0};
  uint64_t fAdePTSeed{1234567};
  float fHitBufferFlushThreshold{0.8};
  float fCPUCapacityFactor{2.5};
  double fHitBufferSafetyFactor{1.5};
  double fMillionsOfTrackSlots{1};
  double fMillionsOfHitSlots{1};
  unsigned short fLastNParticlesOnCPU{0};
  unsigned short fMaxWDTIter{5};
  double fWDTKineticEnergyLimit{0.2}; // 200 keV

  std::vector<std::string> fGPURegionNames{};
  std::vector<std::string> fCPURegionNames{};
  std::vector<std::string> fWDTRegionNames{};
  std::vector<std::string> fDeadRegionNames{};

  std::string fVecGeomGDML{""};
  std::string fCovfieBfieldFile{""};
  std::unique_ptr<AdePTConfigurationMessenger> fAdePTConfigurationMessenger;
};

#endif
