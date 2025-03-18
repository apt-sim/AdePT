// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

///   4HepEmTrackingManagerSpecialized class:
///   The derived class from G4HepEmTrackingManager must implement HandOverOneTrack and CheckEarlyTrackingExit to allow
///   for an early exit of the tracking loop in the G4HepEmTrackingManager

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
  /// @brief Set whether AdePT should transport particles across the whole geometry
  void SetTrackInAllRegions(bool trackInAllRegions) { fTrackInAllRegions = trackInAllRegions; }
  bool GetTrackInAllRegions() const { return fTrackInAllRegions; }

  // Implement HandOverTrack that returns the track if it ends up in the GPU region
  void HandOverOneTrack(G4Track *aTrack) override;

  // Implement the early tracking exit function
  bool CheckEarlyTrackingExit(G4Track *track, G4EventManager *evtMgr, G4UserTrackingAction *userTrackingAction,
                              G4TrackVector &secondaries) const override;

  void ResetFinishEventOnCPUSize(int num_threads) { fFinishEventOnCPU.resize(num_threads, -1); }

  void SetFinishEventOnCPU(int threadid, int eventid) { fFinishEventOnCPU[threadid] = eventid; }

  int GetFinishEventOnCPU(int threadid) { return fFinishEventOnCPU[threadid]; }

private:
  bool fTrackInAllRegions = false;          ///< Whether the whole geometry is a GPU region
  std::set<G4Region const *> fGPURegions{}; ///< List of GPU regions
  std::vector<int> fFinishEventOnCPU;       ///< vector over number of threads to keep certain leaked tracks on GPU

  // G4Region const * fPreviousRegion = nullptr;
};

#endif // G4HepEmTrackingManagerSpecialized_h
