// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

/// \brief Implementation of the TrackingAction class

#include "TrackingAction.hh"
#include "Run.hh"
#include "TrackLineageInfo.hh"
#include "TruthHistogrammer.hh"

#include "G4RunManager.hh"
#include "G4Track.hh"

namespace {

Run *CurrentRun()
{
  auto *runManager = G4RunManager::GetRunManager();
  return runManager != nullptr ? static_cast<Run *>(runManager->GetNonConstCurrentRun()) : nullptr;
}

} // namespace

TrackingAction::TrackingAction() : G4UserTrackingAction() {}

void TrackingAction::PreUserTrackingAction(const G4Track *track)
{
  auto *run               = CurrentRun();
  auto *truthHistogrammer = run != nullptr ? run->GetTruthHistogrammer() : nullptr;
  auto *mutableTrack      = const_cast<G4Track *>(track);
  auto *trackLineageInfo  = track != nullptr ? static_cast<TrackLineageInfo *>(track->GetUserInformation()) : nullptr;

  // Primaries enter tracking without a parent, so this callback seeds the root
  // of the shower tree exactly once. GPU-born secondaries already receive their
  // lineage from the parent step in SteppingAction before they are tracked.
  if (trackLineageInfo == nullptr && mutableTrack != nullptr && mutableTrack->GetParentID() == 0 &&
      mutableTrack->GetUserInformation() == nullptr) {
    trackLineageInfo = new TrackLineageInfo(mutableTrack->GetTrackID(), 0u);
    mutableTrack->SetUserInformation(trackLineageInfo);
  }

  if (truthHistogrammer == nullptr || trackLineageInfo == nullptr || trackLineageInfo->HasRecordedInitial()) return;

  // Record the initial snapshot once per track after the lineage is in place.
  // For secondaries this callback now runs after the parent step attached the
  // propagated primary ancestor and generation.
  truthHistogrammer->RecordPrimaryAncestorPopulation(trackLineageInfo->GetPrimaryTrackID());
  truthHistogrammer->RecordGenerationPopulation(trackLineageInfo->GetGeneration());
  truthHistogrammer->RecordInitialTrack(track);
  trackLineageInfo->MarkRecordedInitial();
}

void TrackingAction::PostUserTrackingAction(const G4Track *track)
{
  auto *run               = CurrentRun();
  auto *truthHistogrammer = run != nullptr ? run->GetTruthHistogrammer() : nullptr;
  auto *trackLineageInfo  = track != nullptr ? static_cast<TrackLineageInfo *>(track->GetUserInformation()) : nullptr;

  if (truthHistogrammer != nullptr && trackLineageInfo != nullptr) {
    const auto *step = track != nullptr ? track->GetStep() : nullptr;
    if (step != nullptr && step->GetPreStepPoint() != nullptr && step->GetPostStepPoint() != nullptr) {
      // Final snapshots are recorded from the terminal tracking callback once
      // Geant4 has attached the complete last step to the track.
      truthHistogrammer->RecordFinalTrack(track);
    }
  }
}
