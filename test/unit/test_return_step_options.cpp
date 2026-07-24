// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/AdePTConfiguration.hh>

#include <G4StateManager.hh>
#include <G4UIcommand.hh>
#include <G4UIcommandTree.hh>
#include <G4UImanager.hh>
#include <gtest/gtest.h>

#include <algorithm>

namespace {
class G4StateGuard {
public:
  explicit G4StateGuard(G4StateManager *stateManager)
      : fStateManager(stateManager), fOriginalState(stateManager->GetCurrentState())
  {
  }

  ~G4StateGuard() { fStateManager->SetNewState(fOriginalState); }

private:
  G4StateManager *fStateManager;
  G4ApplicationState fOriginalState;
};
} // namespace

TEST(AdePTConfiguration, IdleReturnStepCommandsFreezeWithTransportSnapshot)
{
  AdePTConfiguration configuration;
  auto *stateManager = G4StateManager::GetStateManager();
  G4StateGuard stateGuard(stateManager);
  ASSERT_TRUE(stateManager->SetNewState(G4State_Idle));

  auto *uiManager = G4UImanager::GetUIpointer();
  ASSERT_NE(uiManager->GetTree()->FindPath("/adept/returnFirstAndLastStep"), nullptr);
  ASSERT_NE(uiManager->GetTree()->FindPath("/adept/returnAllSteps"), nullptr);
  auto *returnFirstAndLast = uiManager->GetTree()->FindPath("/adept/returnFirstAndLastStep");
  auto *returnAll          = uiManager->GetTree()->FindPath("/adept/returnAllSteps");
  EXPECT_TRUE(returnFirstAndLast->ToBeBroadcasted());
  EXPECT_TRUE(returnAll->ToBeBroadcasted());

  // With no AvailableForStates restriction, Geant4 leaves the commands
  // available in every application state. The lifecycle guard below decides
  // whether their values can still be changed.
  for (const auto state :
       {G4State_PreInit, G4State_Init, G4State_Idle, G4State_GeomClosed, G4State_EventProc, G4State_Abort}) {
    EXPECT_NE(std::find(returnFirstAndLast->GetStateList()->begin(), returnFirstAndLast->GetStateList()->end(), state),
              returnFirstAndLast->GetStateList()->end());
    EXPECT_NE(std::find(returnAll->GetStateList()->begin(), returnAll->GetStateList()->end(), state),
              returnAll->GetStateList()->end());
  }

  // This models Gauss after master initialization but before the first worker
  // constructs AdePTTransport. The Idle commands must still configure the
  // values that will be copied into the GPU worker.
  EXPECT_EQ(uiManager->ApplyCommand("/adept/returnFirstAndLastStep true"), 0);
  EXPECT_EQ(uiManager->ApplyCommand("/adept/returnAllSteps true"), 0);
  EXPECT_TRUE(configuration.GetReturnFirstAndLastStep());
  EXPECT_TRUE(configuration.GetReturnAllSteps());

  // This is the point at which AdePTTransport takes its by-value kernel-option
  // snapshot. Later Idle commands may repeat, but must not change, the values.
  configuration.LockReturnStepOptions();
  EXPECT_TRUE(configuration.ReturnStepOptionsAreLocked());
  EXPECT_EQ(uiManager->ApplyCommand("/adept/returnFirstAndLastStep true"), 0);
  EXPECT_EQ(uiManager->ApplyCommand("/adept/returnAllSteps true"), 0);
  EXPECT_NE(uiManager->ApplyCommand("/adept/returnFirstAndLastStep false"), 0);
  EXPECT_NE(uiManager->ApplyCommand("/adept/returnAllSteps false"), 0);

  EXPECT_TRUE(configuration.GetReturnFirstAndLastStep());
  EXPECT_TRUE(configuration.GetReturnAllSteps());
}
