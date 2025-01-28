// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"
#include "G4EventManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"

class SteppingAction final : public G4UserSteppingAction {

public:
  SteppingAction(int numSteps) : fNumSteps{numSteps} {}
  ~SteppingAction() = default;
  void SetNumSteps(int aNumSteps) { fNumSteps = aNumSteps; }

  /// Kill the particle if it has done too many steps
  void UserSteppingAction(const G4Step *theStep) override
  {
    G4Track *track = theStep->GetTrack();
    if (track->GetCurrentStepNumber() >= fNumSteps && track->GetParentID() > 9000000) {
      if (const auto eventId = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
          eventId != fLastEventID) {
        fLastEventID = eventId;
        fTotalEnergy = 0.;
      }
      fTotalEnergy += track->GetKineticEnergy();
      G4cout << "Warning: Killing track over " << fNumSteps << " steps: E: " << track->GetKineticEnergy() / MeV
             << "MeV pdg=" << track->GetDynamicParticle()->GetPDGcode()
             << " total energy killed: " << fTotalEnergy / GeV << " GeV" << G4endl;
      track->SetTrackStatus(fStopAndKill);
    }
  }

private:
  double fTotalEnergy{0.};
  int fLastEventID{0};
  int fNumSteps{0};
};

#endif
