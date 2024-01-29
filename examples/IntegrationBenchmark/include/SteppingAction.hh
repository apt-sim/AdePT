// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4RegionStore.hh"
#include "G4Region.hh"

class DetectorConstruction;
class RunAction;
class TrackingAction;

class SteppingAction : public G4UserSteppingAction {

public:
  SteppingAction(TrackingAction *aTrackingAction, bool aDoBenchmark);
  ~SteppingAction() override;
  void UserSteppingAction(const G4Step *step) override;
  void SetNumSteps(int aNumSteps) { fNumSteps = aNumSteps; }

private:
  TrackingAction *fTrackingAction;
  int fNumSteps{0};
  bool fDoBenchmark{false};
};

#endif