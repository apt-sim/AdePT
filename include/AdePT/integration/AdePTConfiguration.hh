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

  std::string GetGPURegionName() { return fGPURegionName; }
  bool IsAdePTActivated() { return fAdePTActivated; }
  int GetVerbosity() { return fVerbosity; };
  int GetTransportBufferThreshold() { return fTransportBufferThreshold; }
  double GetMillionsOfTrackSlots() { return fMillionsOfTrackSlots; }
  double GetMillionsOfHitSlots() { return fMillionsOfHitSlots; }
  float GetHitBufferFlushThreshold() { return fHitBufferFlushThreshold; }

private:
  int fRandomSeed;
  std::string fGPURegionName;
  bool fAdePTActivated;
  int fVerbosity;
  int fTransportBufferThreshold;
  double fMillionsOfTrackSlots;
  double fMillionsOfHitSlots;
  float fHitBufferFlushThreshold;

  AdePTConfigurationMessenger *fAdePTConfigurationMessenger;
};