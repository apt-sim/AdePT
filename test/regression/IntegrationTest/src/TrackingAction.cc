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
/// \file electromagnetic/TestEm1/src/TrackingAction.cc
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

TrackLineageInfo *GetTrackLineageInfo(const G4Track *track)
{
  return track != nullptr ? dynamic_cast<TrackLineageInfo *>(track->GetUserInformation()) : nullptr;
}

} // namespace

TrackingAction::TrackingAction() : G4UserTrackingAction() {}

void TrackingAction::PreUserTrackingAction(const G4Track *track)
{
  auto *run               = CurrentRun();
  auto *truthHistogrammer = run != nullptr ? run->GetTruthHistogrammer() : nullptr;
  auto *mutableTrack      = const_cast<G4Track *>(track);
  auto *trackLineageInfo  = GetTrackLineageInfo(track);

  if (trackLineageInfo == nullptr && mutableTrack != nullptr && mutableTrack->GetParentID() == 0 &&
      mutableTrack->GetUserInformation() == nullptr) {
    trackLineageInfo = new TrackLineageInfo(mutableTrack->GetTrackID(), 0u);
    mutableTrack->SetUserInformation(trackLineageInfo);
  }

  if (trackLineageInfo == nullptr && mutableTrack != nullptr && mutableTrack->GetParentID() != 0 &&
      mutableTrack->GetUserInformation() == nullptr) {
    // For returned GPU secondaries, PreUserTrackingAction can still happen
    // before the parent UserSteppingAction. Seed a provisional lineage here;
    // SteppingAction will overwrite it once the resolved parent lineage is known.
    trackLineageInfo = new TrackLineageInfo(mutableTrack->GetParentID(), 1u);
    mutableTrack->SetUserInformation(trackLineageInfo);
  }

  if (truthHistogrammer == nullptr || trackLineageInfo == nullptr || trackLineageInfo->HasRecordedInitial()) return;

  truthHistogrammer->RecordPrimaryAncestorPopulation(trackLineageInfo->GetPrimaryTrackID());
  truthHistogrammer->RecordGenerationPopulation(trackLineageInfo->GetGeneration());
  truthHistogrammer->RecordInitialTrack(track);
  trackLineageInfo->MarkRecordedInitial();
}

void TrackingAction::PostUserTrackingAction(const G4Track *track)
{
  auto *run               = CurrentRun();
  auto *truthHistogrammer = run != nullptr ? run->GetTruthHistogrammer() : nullptr;
  auto *trackLineageInfo  = GetTrackLineageInfo(track);

  if (truthHistogrammer != nullptr && trackLineageInfo != nullptr) {
    const auto *step = track != nullptr ? track->GetStep() : nullptr;
    if (step != nullptr && step->GetPreStepPoint() != nullptr && step->GetPostStepPoint() != nullptr &&
        !trackLineageInfo->HasRecordedFinal()) {
      truthHistogrammer->RecordFinalTrack(track);
      trackLineageInfo->MarkRecordedFinal();
    }
  }
}
