// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/AdePTConfiguration.hh>

#include <G4StateManager.hh>
#include <G4UImanager.hh>
#include <gtest/gtest.h>

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
  uiManager->ApplyCommand("/adept/returnFirstAndLastStep false");
  uiManager->ApplyCommand("/adept/returnAllSteps false");

  EXPECT_TRUE(configuration.GetReturnFirstAndLastStep());
  EXPECT_TRUE(configuration.GetReturnAllSteps());
}
