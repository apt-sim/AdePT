// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/G4HepEmTrackingManagerSpecialized.hh>

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"

G4HepEmTrackingManagerSpecialized::G4HepEmTrackingManagerSpecialized() : G4HepEmTrackingManager() {}

G4HepEmTrackingManagerSpecialized::~G4HepEmTrackingManagerSpecialized() {}

/// @brief The function checks within the TrackElectron and TrackGamma calls in G4HepEmTracking manager, if a barrier is
/// hit.
// Here is the specialized AdePT G4HepEmTrackingManager implementation that if the track enters a GPU region, it is
// stopped on CPU and subsequently tracked on GPU
/// @param track a G4track to be checked
/// @param evtMgr the G4EventManager to stack secondaries
/// @param userTrackingAction the UserTrackingAction
/// @return
bool G4HepEmTrackingManagerSpecialized::CheckEarlyTrackingExit(G4Track *track, G4EventManager *evtMgr,
                                                               G4UserTrackingAction *userTrackingAction,
                                                               G4TrackVector &secondaries) const
{

  // GetNextVolume since when this function is called in TrackElectron/Gamma in G4HepEmTrackingManager
  // the current volume is not yet updated
  G4Region const *region = track->GetNextVolume()->GetLogicalVolume()->GetRegion();

  G4int threadId = G4Threading::G4GetThreadId();

  // TODO: for more efficient use, we only have to check for region within GPURegions, if the region changed.
  //       This can be checked from the pre- and post-steppoint

  // Not in the GPU region, continue normal tracking with G4HepEmTrackingManager
  if ((!GetTrackInAllRegions() && fGPURegions.find(region) == fGPURegions.end()) || fFinishEventOnCPU[threadId] > 0) {
    return false; // Continue tracking with G4HepEmTrackingManager
  } else {

    // Track entered a GPU region. Now, the track must be properly ended here in the same way as G4HepEm would,
    // since G4HepEm exists the TrackElectron / TrackGamma function immediately after this function returns true.
    // This includes Ending the tracking for fast simulation manager, calling the UserTrackingAction,
    // deleting the trajectory, and stacking the secondaries

    // Invoke the fast simulation manager process EndTracking interface (if any)
    const G4ParticleDefinition *part = track->GetParticleDefinition();

    G4VProcess *fFastSimProc;
    if (part == G4Electron::Definition()) {
      fFastSimProc = fFastSimProcess[0];
    } else if (part == G4Positron::Definition()) {
      fFastSimProc = fFastSimProcess[1];
    } else if (part == G4Gamma::Definition()) {
      fFastSimProc = fFastSimProcess[2];
    } else {
      throw std::runtime_error("Unexpected particle type!");
    }

    if (fFastSimProc != nullptr) {
      fFastSimProc->EndTracking();
    }

    // call PostUserTrackingAction
    if (userTrackingAction) {
      userTrackingAction->PostUserTrackingAction(track);
    }

    // // Delete the trajectory object (if the user set any)
    G4VTrajectory *theTrajectory = evtMgr->GetTrackingManager()->GetStoreTrajectory() == 0
                                       ? nullptr
                                       : evtMgr->GetTrackingManager()->GimmeTrajectory();
    if (theTrajectory != nullptr) {
      delete theTrajectory;
    }

    // Push secondaries
    evtMgr->StackTracks(&secondaries);

    // return true to stop tracking in G4HepEmTrackingManager to hand over to GPU
    return true;
  }
}

void G4HepEmTrackingManagerSpecialized::HandOverOneTrack(G4Track *aTrack)
{
  const G4ParticleDefinition *part = aTrack->GetParticleDefinition();

  // Do tracking with G4HepEm and check whether the track was finished
  bool tracking_finished = true;
  if (part == G4Electron::Definition() || part == G4Positron::Definition()) {
    tracking_finished = TrackElectron(aTrack);
  } else if (part == G4Gamma::Definition()) {
    tracking_finished = TrackGamma(aTrack);
  }

  // if G4HepEm finished the track, set status to StopAndKill, the track will be deleted in AdePTTrackingManager
  // otherwise, it is kept to be transported on GPU
  if (tracking_finished) {
    aTrack->SetTrackStatus(fStopAndKill);
  }
}
