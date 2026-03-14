// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "SteppingAction.hh"
#include "Run.hh"
#include "TrackLineageInfo.hh"
#include "TruthHistogrammer.hh"

#include "G4RunManager.hh"
#include "G4Step.hh"

namespace {

/// Helper to access the mutable regression run object from Geant4 callbacks.
Run *CurrentRun()
{
  auto *runManager = G4RunManager::GetRunManager();
  return runManager != nullptr ? static_cast<Run *>(runManager->GetNonConstCurrentRun()) : nullptr;
}

} // namespace

SteppingAction::SteppingAction() : G4UserSteppingAction() {}

SteppingAction::~SteppingAction() {}

void SteppingAction::UserSteppingAction(const G4Step *theStep)
{
  if (theStep != nullptr) {
    auto *run               = CurrentRun();
    auto *truthHistogrammer = run != nullptr ? run->GetTruthHistogrammer() : nullptr;
    if (truthHistogrammer != nullptr) {
      truthHistogrammer->RecordStep(theStep);
    }

    auto *parentTrack = theStep->GetTrack();
    auto *parentInfo =
        parentTrack != nullptr ? dynamic_cast<TrackLineageInfo *>(parentTrack->GetUserInformation()) : nullptr;

    const auto *secondaries = theStep->GetSecondary();
    if (secondaries != nullptr && parentInfo != nullptr) {
      const int primaryTrackID      = parentInfo->GetPrimaryTrackID();
      const unsigned int generation = parentInfo->GetGeneration() + 1u;

      for (auto *secondary : *secondaries) {
        if (secondary == nullptr) continue;
        auto *secondaryInfo = dynamic_cast<TrackLineageInfo *>(secondary->GetUserInformation());
        if (secondaryInfo == nullptr) {
          if (secondary->GetUserInformation() != nullptr) continue;
          secondaryInfo = new TrackLineageInfo(primaryTrackID, generation);
          secondary->SetUserInformation(secondaryInfo);
        } else {
          // TrackingAction may already have attached a provisional fallback
          // lineage. Replace it with the lineage derived from the resolved
          // parent step now that the parent is available.
          secondaryInfo->SetLineage(primaryTrackID, generation);
        }

        if (truthHistogrammer != nullptr && secondaryInfo != nullptr && !secondaryInfo->HasRecordedInitial()) {
          truthHistogrammer->RecordPrimaryAncestorPopulation(secondaryInfo->GetPrimaryTrackID());
          truthHistogrammer->RecordGenerationPopulation(secondaryInfo->GetGeneration());
          truthHistogrammer->RecordInitialTrack(secondary);
          secondaryInfo->MarkRecordedInitial();
        }
      }
    }
  }

  // Kill the particle if it has done over 1000 steps
  auto *track = theStep != nullptr ? theStep->GetTrack() : nullptr;
  if (track != nullptr && track->GetCurrentStepNumber() > 10000) {
    G4cout << "Warning: Killing track over 10000 steps" << G4endl;
    track->SetTrackStatus(fStopAndKill);
  }
}
