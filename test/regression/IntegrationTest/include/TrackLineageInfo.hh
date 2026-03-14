// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACKLINEAGEINFO_HH
#define TRACKLINEAGEINFO_HH

#include "G4VUserTrackInformation.hh"

/**
 * @brief Minimal lineage payload attached to regression-test tracks.
 *
 * The drift test only needs two stable lineage properties:
 * - the primary ancestor track ID
 * - the shower generation
 *
 * The boolean flags keep the histogramming hooks idempotent even when a track
 * passes through multiple callbacks on the AdePT/G4 boundary.
 */
class TrackLineageInfo : public G4VUserTrackInformation {
public:
  TrackLineageInfo(int primaryTrackID, unsigned int generation)
      : fPrimaryTrackID(primaryTrackID), fGeneration(generation)
  {
  }

  ~TrackLineageInfo() override = default;

  void Print() const override {}

  int GetPrimaryTrackID() const { return fPrimaryTrackID; }
  unsigned int GetGeneration() const { return fGeneration; }
  /// Overwrite the provisional lineage once the parent step is known.
  void SetLineage(int primaryTrackID, unsigned int generation)
  {
    fPrimaryTrackID = primaryTrackID;
    fGeneration     = generation;
  }

  bool HasRecordedInitial() const { return fInitialRecorded; }
  void MarkRecordedInitial() { fInitialRecorded = true; }

  bool HasRecordedFinal() const { return fFinalRecorded; }
  void MarkRecordedFinal() { fFinalRecorded = true; }

private:
  int fPrimaryTrackID;
  unsigned int fGeneration;
  bool fInitialRecorded{false};
  bool fFinalRecorded{false};
};

#endif
