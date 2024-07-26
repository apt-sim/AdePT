// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_CONFIGURATION_HH
#define ADEPT_CONFIGURATION_HH

#include <AdePT/integration/AdePTConfigurationMessenger.hh>
#include <AdePT/core/AdePTTransportInterface.hh>

#include <memory>
#include <string>
#include <vector>

class AdePTConfiguration {
public:
  AdePTConfiguration();
  ~AdePTConfiguration();
  void SetRandomSeed(int randomSeed) { fRandomSeed = randomSeed; }
  void SetTrackInAllRegions(bool trackInAllRegions) { fTrackInAllRegions = trackInAllRegions; }
  void AddGPURegionName(std::string name) { fGPURegionNames.push_back(name); }
  void SetAdePTActivation(bool activateAdePT) { fAdePTActivated = activateAdePT; }
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; };
  void SetTransportBufferThreshold(int threshold) { fTransportBufferThreshold = threshold; }
  void SetMillionsOfTrackSlots(double millionSlots) { fMillionsOfTrackSlots = millionSlots; }
  void SetMillionsOfHitSlots(double millionSlots) { fMillionsOfHitSlots = millionSlots; }
  void SetHitBufferFlushThreshold(float threshold) { fHitBufferFlushThreshold = threshold; }

  // We temporarily load VecGeom geometry from GDML
  void SetVecGeomGDML(std::string filename) { fVecGeomGDML = filename; }

  bool GetTrackInAllRegions() { return fTrackInAllRegions; }
  std::vector<std::string> *GetGPURegionNames() { return &fGPURegionNames; }
  bool IsAdePTActivated() { return fAdePTActivated; }
  int GetVerbosity() { return fVerbosity; };
  int GetTransportBufferThreshold() { return fTransportBufferThreshold; }
  double GetMillionsOfTrackSlots() { return fMillionsOfTrackSlots; }
  double GetMillionsOfHitSlots() { return fMillionsOfHitSlots; }
  float GetHitBufferFlushThreshold() { return fHitBufferFlushThreshold; }

  std::shared_ptr<AdePTTransportInterface> CreateAdePTInstance(unsigned int nThread);

  using AdePTFactoryFunc_t = std::shared_ptr<AdePTTransportInterface> (*)(
      unsigned int nThread, unsigned int nTrackSlot, unsigned int nHitSlot, int verbosity,
      std::vector<std::string> const *GPURegionNames, bool trackInAllRegions);
  static void SetAdePTFactoryFunction(AdePTFactoryFunc_t func) { fAdePTFactoryFunction = func; }

  // Temporary
  std::string GetVecGeomGDML() { return fVecGeomGDML; }

private:
  int fRandomSeed;
  bool fTrackInAllRegions{false};
  std::vector<std::string> fGPURegionNames{};
  bool fAdePTActivated{true};
  int fVerbosity{0};
  int fTransportBufferThreshold{200};
  double fMillionsOfTrackSlots{1};
  double fMillionsOfHitSlots{1};
  float fHitBufferFlushThreshold{0.8};
  int fNThread = -1;

  std::string fVecGeomGDML{""};
  static inline AdePTFactoryFunc_t fAdePTFactoryFunction = nullptr;

  std::unique_ptr<AdePTConfigurationMessenger> fAdePTConfigurationMessenger;
};

#endif