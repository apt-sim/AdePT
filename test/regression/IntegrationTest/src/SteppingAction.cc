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

    const auto *secondaries = theStep->GetSecondaryInCurrentStep();
    if (secondaries != nullptr && parentInfo != nullptr) {
      // This is the only callback where Geant4 exposes the stepped parent and
      // the newly created secondaries together. The drift test therefore
      // attaches the MC-truth lineage here and propagates it one generation
      // deeper into the shower tree.
      const int primaryTrackID      = parentInfo->GetPrimaryTrackID();
      const unsigned int generation = parentInfo->GetGeneration() + 1u;

      for (auto *secondary : *secondaries) {
        if (secondary == nullptr) continue;
        auto *secondaryInfo = static_cast<TrackLineageInfo *>(secondary->GetUserInformation());
        if (secondaryInfo == nullptr) {
          secondaryInfo = new TrackLineageInfo(primaryTrackID, generation);
          secondary->SetUserInformation(secondaryInfo);
        } else {
          // Keep the lineage payload in sync if the same secondary is observed
          // again while the parent step is still being processed.
          secondaryInfo->SetLineage(primaryTrackID, generation);
        }

        if (truthHistogrammer != nullptr && secondaryInfo != nullptr && !secondaryInfo->HasRecordedInitial()) {
          // The initial snapshot is recorded when the secondary first becomes
          // fully linked to its parent lineage. Mark it so later callbacks do
          // not duplicate the same track in the truth histograms.
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
