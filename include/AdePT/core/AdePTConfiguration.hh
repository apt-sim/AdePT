#include <string>
#include <vector>
#include "AdePTConfigurationMessenger.hh"

class AdePTConfiguration {
public:
  AdePTConfiguration() { fAdePTConfigurationMessenger = new AdePTConfigurationMessenger(this); }
  ~AdePTConfiguration() { delete fAdePTConfigurationMessenger; }
  void SetRandomSeed(int randomSeed) { fRandomSeed = randomSeed; }
  void SetGPURegionName(std::string name) { fGPURegionName = name; }
  void SetAdePTActivation(bool activateAdePT) { fAdePTActivated = activateAdePT; }
  void SetVerbosity(int verbosity) { fVerbosity = verbosity; };
  void SetTransportBufferThreshold(int threshold) { fTransportBufferThreshold = threshold; }
  void SetMillionsOfTrackSlots(double millionSlots) { fMillionsOfTrackSlots = millionSlots; }
  void SetMillionsOfHitSlots(double millionSlots) { fMillionsOfHitSlots = millionSlots; }
  void SetHitBufferFlushThreshold(float threshold) { fHitBufferFlushThreshold = threshold; }

  // We temporarily load VecGeom geometry from GDML
  void SetVecGeomGDML(std::string filename) { fVecGeomGDML = filename; }

  std::string GetGPURegionName() { return fGPURegionName; }
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
  std::string fGPURegionName{""};
  bool fAdePTActivated{true};
  int fVerbosity{0};
  int fTransportBufferThreshold{200};
  double fMillionsOfTrackSlots{1};
  double fMillionsOfHitSlots{1};
  float fHitBufferFlushThreshold{0.8};

  std::string fVecGeomGDML{""};

  AdePTConfigurationMessenger *fAdePTConfigurationMessenger;
};