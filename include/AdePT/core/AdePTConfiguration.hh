// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>
#include <AdePT/integration/AdePTConfigurationMessenger.hh>

class AdePTConfiguration {
public:
  AdePTConfiguration() { fAdePTConfigurationMessenger = new AdePTConfigurationMessenger(this); }
  ~AdePTConfiguration() { delete fAdePTConfigurationMessenger; }
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

  // Temporary
  std::string GetVecGeomGDML(){ return fVecGeomGDML; }

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

  std::string fVecGeomGDML{""};

  AdePTConfigurationMessenger *fAdePTConfigurationMessenger;
};