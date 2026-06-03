// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "G4VUserTrackInformation.hh"

/**
 * @brief Minimal lineage payload attached to regression-test tracks.
 *
 * The drift test only needs two stable lineage properties:
 * - the primary ancestor track ID
 * - the shower generation
 *
 * Lineage is assigned once. The initial-recorded flag avoids double counting secondaries: they are first
 * seen in the parent stepping callback and then enter their own tracking
 * callback afterwards.
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

  bool HasRecordedInitial() const { return fInitialRecorded; }
  void MarkRecordedInitial() { fInitialRecorded = true; }

private:
  const int fPrimaryTrackID;
  const unsigned int fGeneration;
  bool fInitialRecorded{false};
};
