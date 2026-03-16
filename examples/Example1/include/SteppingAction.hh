// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"

class SteppingAction : public G4UserSteppingAction {

public:
  SteppingAction();
  ~SteppingAction() override;
  void UserSteppingAction(const G4Step *step) override;
};

#endif
