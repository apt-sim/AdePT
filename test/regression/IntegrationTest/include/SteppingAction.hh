// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef STEPPINGACTION_HH
#define STEPPINGACTION_HH

#include "G4UserSteppingAction.hh"

/**
 * @brief Regression-test stepping hooks.
 *
 * In ROOT truth mode this action records per-step observables and refines the
 * lineage of secondaries once the parent step is available. It also keeps the
 * legacy protection against runaway tracks with excessive step counts.
 */
class SteppingAction : public G4UserSteppingAction {

public:
  SteppingAction();
  ~SteppingAction() override;
  void UserSteppingAction(const G4Step *step) override;
};

#endif
