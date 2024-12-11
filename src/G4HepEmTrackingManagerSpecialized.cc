// SPDX-FileCopyrightText: 2023 CERN
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

  G4Region const *region = track->GetVolume()->GetLogicalVolume()->GetRegion();

  // TODO: for more efficient use, we only have to check for region within GPURegions, if the region changed.
  //       This can be checked from the pre- and post-steppoint

  // Not in the GPU region, continue normal tracking with G4HepEmTrackingManager
  if (fGPURegions.find(region) != fGPURegions.end()) {
    return false; // Continue tracking with G4HepEmTrackingManager
  } else {

    // first: end tracking for previous particle

    // FIXME ignoring FastSimProc for now
    // Invoke the fast simulation manager process EndTracking interface (if any)
    // if (fFastSimProc != nullptr) {
    //   fFastSimProc->EndTracking();
    // }

    if (userTrackingAction) {
      userTrackingAction->PostUserTrackingAction(track);
    }

    // FIXME ignore the trajectory for now
    // // Delete the trajectory object (if the user set any)
    // if (theTrajectory != nullptr) {
    //   delete theTrajectory;
    // }

    evtMgr->StackTracks(&secondaries);

    return true; // stop tracking to hand over to GPU
  }
}

void G4HepEmTrackingManagerSpecialized::HandOverOneTrack(G4Track *aTrack)
{
  const G4ParticleDefinition *part = aTrack->GetParticleDefinition();

  bool tracking_finished = true; // whether G4HepEm finishes the track itself
  if (part == G4Electron::Definition() || part == G4Positron::Definition()) {
    tracking_finished = TrackElectron(aTrack);
  } else if (part == G4Gamma::Definition()) {
    tracking_finished = TrackGamma(aTrack);
  }

  // if G4HepEm finished the track, it can be deleted, otherwise it is kept to be transported on GPU
  if (tracking_finished) {
    aTrack->SetTrackStatus(fStopAndKill);
    delete aTrack;
  }
}
