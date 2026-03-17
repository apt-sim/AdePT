// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

/// \brief Definition of the TrackingAction class

#ifndef TRACKINGACTION_HH
#define TRACKINGACTION_HH

#include "G4UserTrackingAction.hh"

/**
 * @brief Regression-test tracking hooks.
 *
 * The ROOT truth mode uses the tracking callbacks to capture the initial and
 * final track snapshots. Primaries receive their lineage here, while returned
 * GPU secondaries arrive with lineage already attached by the parent step in
 * the stepping action.
 */
class TrackingAction : public G4UserTrackingAction {

public:
  TrackingAction();
  ~TrackingAction() override = default;

  void PreUserTrackingAction(const G4Track *) override;
  void PostUserTrackingAction(const G4Track *) override;
};

#endif // TRACKINGACTION_HH
