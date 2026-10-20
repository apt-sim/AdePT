// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <AdePT/g4integration/CMSSteppingActionConfig.hh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class AdePTConfigurationMessenger;

/// @brief Create and configure instances of an AdePT transport implementation.
///
class AdePTConfiguration {
public:
  AdePTConfiguration();
  ~AdePTConfiguration();
  void SetNumThreads(int numThreads) { fNumThreads = numThreads; }
  bool SetTrackInAllRegions(bool trackInAllRegions)
  {
    if (sTransportInitializationOptionsLocked) return fTrackInAllRegions == trackInAllRegions;
    fTrackInAllRegions = trackInAllRegions;
    return true;
  }
  void SetCallUserActions(bool callUserActions) { fCallUserActions = callUserActions; }
  void SetCallUserSteppingAction(bool callUserSteppingAction) { fCallUserSteppingAction = callUserSteppingAction; }
  void SetCallUserTrackingAction(bool callUserTrackingAction) { fCallUserTrackingAction = callUserTrackingAction; }
  bool SetReturnAllSteps(bool returnAllSteps)
  {
    if (sTransportInitializationOptionsLocked) return fReturnAllSteps == returnAllSteps;
    fReturnAllSteps = returnAllSteps;
    return true;
  }
  bool SetReturnFirstAndLastStep(bool returnFirstAndLastStep)
  {
    if (sTransportInitializationOptionsLocked) {
      return fReturnFirstAndLastStep == returnFirstAndLastStep;
    }
    fReturnFirstAndLastStep = returnFirstAndLastStep;
    return true;
  }
  void LockTransportInitializationOptions() { sTransportInitializationOptionsLocked = true; }
  bool TransportInitializationOptionsAreLocked() const { return sTransportInitializationOptionsLocked; }
  // Keep the old names for source compatibility.
  void LockReturnStepOptions() { LockTransportInitializationOptions(); }
  bool ReturnStepOptionsAreLocked() const { return TransportInitializationOptionsAreLocked(); }
  bool AddGPURegionName(std::string name)
  {
    if (sTransportInitializationOptionsLocked) {
      return std::find(fGPURegionNames.begin(), fGPURegionNames.end(), name) != fGPURegionNames.end();
    }
    fGPURegionNames.push_back(std::move(name));
    return true;
  }
  bool RemoveGPURegionName(std::string name)
  {
    if (sTransportInitializationOptionsLocked) {
      return std::find(fCPURegionNames.begin(), fCPURegionNames.end(), name) != fCPURegionNames.end();
    }
    fCPURegionNames.push_back(std::move(name));
    return true;
  }
  void AddWDTRegionName(std::string name) { fWDTRegionNames.push_back(name); }
  void AddDeadRegionName(std::string name) { fCMSSteppingActionConfig.deadRegions.push_back(name); }
  /// @brief Replace the configuration of the CMS stepping action, including dead regions added with AddDeadRegionName.
  void SetCMSSteppingActionConfig(CMSSteppingActionConfig config) { fCMSSteppingActionConfig = std::move(config); }
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
  void SetMaxChargedLooperCount(int maxCount)
  {
    constexpr int maxActiveChargedLooperCount = std::numeric_limits<unsigned short>::max() - 1000;
    if (maxCount < 0 || maxCount > maxActiveChargedLooperCount) {
      throw std::out_of_range("MaxChargedLooperCount must be between 0 and 64535");
    }
    fMaxChargedLooperCount = static_cast<unsigned short>(maxCount);
  }
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
  bool GetCallUserActions() const { return fCallUserActions; }
  bool GetCallUserSteppingAction() const { return fCallUserActions || fCallUserSteppingAction; }
  bool GetCallUserTrackingAction() const { return fCallUserActions || fCallUserTrackingAction; }
  bool GetReturnAllSteps() const { return fReturnAllSteps; }
  bool GetReturnFirstAndLastStep() const { return fReturnFirstAndLastStep; }
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
  unsigned short GetMaxChargedLooperCount() const { return fMaxChargedLooperCount; }
  double GetWDTKineticEnergyLimit() const { return fWDTKineticEnergyLimit; }
  float GetHitBufferFlushThreshold() const { return fHitBufferFlushThreshold; }
  float GetCPUCapacityFactor() const { return fCPUCapacityFactor; }
  double GetHitBufferSafetyFactor() const { return fHitBufferSafetyFactor; }
  double GetMillionsOfTrackSlots() const { return fMillionsOfTrackSlots; }
  double GetMillionsOfHitSlots() const { return fMillionsOfHitSlots; }
  const std::vector<std::string> *GetGPURegionNames() const { return &fGPURegionNames; }
  const std::vector<std::string> *GetCPURegionNames() const { return &fCPURegionNames; }
  const std::vector<std::string> &GetWDTRegionNames() const { return fWDTRegionNames; }
  const CMSSteppingActionConfig &GetCMSSteppingActionConfig() const { return fCMSSteppingActionConfig; }

  // Temporary
  std::string GetVecGeomGDML() const { return fVecGeomGDML; }
  std::string GetCovfieBfieldFile() const { return fCovfieBfieldFile; } // todo add #ifdef guards?

private:
  bool fTrackInAllRegions{false};
  bool fCallUserActions{false};
  bool fCallUserSteppingAction{false};
  bool fCallUserTrackingAction{false};
  bool fReturnAllSteps{false};
  bool fReturnFirstAndLastStep{false};
  // These settings cannot change after the first worker initializes AdePT.
  inline static bool sTransportInitializationOptionsLocked{false};
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
  unsigned short fMaxChargedLooperCount{500};
  double fWDTKineticEnergyLimit{0.2}; // 200 keV

  std::vector<std::string> fGPURegionNames{};
  std::vector<std::string> fCPURegionNames{};
  std::vector<std::string> fWDTRegionNames{};
  CMSSteppingActionConfig fCMSSteppingActionConfig{};

  std::string fVecGeomGDML{""};
  std::string fCovfieBfieldFile{""};
  std::unique_ptr<AdePTConfigurationMessenger> fAdePTConfigurationMessenger;
};
