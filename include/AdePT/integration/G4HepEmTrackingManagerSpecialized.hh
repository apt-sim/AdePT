#ifndef G4HepEmTrackingManagerSpecialized_h
#define G4HepEmTrackingManagerSpecialized_h

#include "G4HepEmTrackingManager.hh"

#ifndef G4HepEm_EARLY_TRACKING_EXIT
#error "Build error: G4HepEm must be build with -DG4HepEm_EARLY_TRACKING_EXIT=ON"
#endif

class G4HepEmTrackingManagerSpecialized : public G4HepEmTrackingManager {
public:
  G4HepEmTrackingManagerSpecialized();
  ~G4HepEmTrackingManagerSpecialized();

  void SetGPURegions(const std::set<G4Region const *> &gpuRegions) { fGPURegions = gpuRegions; }

  // Implement HandOverTrack that returns the track if it ends up in the GPU region
  void HandOverOneTrack(G4Track *aTrack) override;

  // Implement the early tracking exit function
  bool CheckEarlyTrackingExit(G4Track *track, G4EventManager *evtMgr, G4UserTrackingAction *userTrackingAction,
                              G4TrackVector &secondaries) const override;

private:
  std::set<G4Region const *> fGPURegions{};
  // G4Region const * fPreviousRegion = nullptr;
};

#endif // G4HepEmTrackingManagerSpecialized_h
