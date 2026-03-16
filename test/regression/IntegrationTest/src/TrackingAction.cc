// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
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
