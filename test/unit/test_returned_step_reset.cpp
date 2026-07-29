// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/returned_steps/ReturnedStepReset.hh>

#include <G4Gamma.hh>

#include <gtest/gtest.h>

TEST(ReturnedStepReset, ClearsOnlyConditionallyAssignedState)
{
  auto *dynamic = new G4DynamicParticle(G4Gamma::Definition(), G4ThreeVector(0., 0., 1.), 12.);
  G4Track track(dynamic, 7., G4ThreeVector(1., 2., 3.));
  track.IncrementCurrentStepNumber();
  track.IncrementCurrentStepNumber();
  track.SetStepLength(4.);
  track.SetTrackStatus(fStopAndKill);
  track.UseGivenVelocity(true);
  track.SetVelocity(5.);
  track.SetBelowThresholdFlag(true);
  track.SetGoodForTrackingFlag(true);

  G4Step step;
  step.SetTrack(&track);
  step.SetStepLength(6.);
  step.SetControlFlag(AvoidHitInvocation);
  step.SetFirstStepFlag();
  step.SetLastStepFlag();
  step.GetPreStepPoint()->SetVelocity(8.);
  step.GetPostStepPoint()->SetVelocity(9.);

  adept::g4integration::detail::ResetForStep(step);

  EXPECT_EQ(track.GetTrackStatus(), fAlive);
  EXPECT_DOUBLE_EQ(track.GetVelocity(), 0.);
  EXPECT_FALSE(step.IsFirstStepInVolume());
  EXPECT_FALSE(step.IsLastStepInVolume());
  EXPECT_DOUBLE_EQ(step.GetPostStepPoint()->GetVelocity(), 0.);

  EXPECT_TRUE(track.UseGivenVelocity());
  EXPECT_TRUE(track.IsBelowThreshold());
  EXPECT_TRUE(track.IsGoodForTracking());
  EXPECT_EQ(step.GetControlFlag(), AvoidHitInvocation);
  EXPECT_DOUBLE_EQ(step.GetPreStepPoint()->GetVelocity(), 8.);
  EXPECT_EQ(track.GetCurrentStepNumber(), 2);
  EXPECT_EQ(track.GetPosition(), G4ThreeVector(1., 2., 3.));
  EXPECT_DOUBLE_EQ(track.GetKineticEnergy(), 12.);
  EXPECT_DOUBLE_EQ(track.GetGlobalTime(), 7.);
  EXPECT_DOUBLE_EQ(track.GetStepLength(), 4.);
  EXPECT_DOUBLE_EQ(step.GetStepLength(), 6.);
}
