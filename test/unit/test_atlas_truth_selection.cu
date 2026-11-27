// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/transport/kernels/AdePTSteppingAction.cuh>
#include <AdePT/transport/support/PhysicalConstants.h>
#include <gtest/gtest.h>

#include "DeviceVisibleMemory.hh"

#include <array>
#include <algorithm>

namespace {
struct Candidate {
  double energy;
  vecgeom::Vector3D<double> direction;
  double mass;
  bool parentHasHostData;
  bool returnLastStep;
  bool expected;
};

__global__ void SelectCandidates(Candidate const *candidates, bool *selected, unsigned int count)
{
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  TrackBase parent;
  parent.hasHostData = candidates[i].parentHasHostData;
  TrackBase child;
  child.eKin = candidates[i].energy;
  child.dir  = candidates[i].direction;
  adept::SteppingAction::SetSecondaryHostData<adept::SteppingAction::ATLASAction>(child, parent, candidates[i].mass,
                                                                                  candidates[i].returnLastStep);
  selected[i] = child.hasHostData;
}
} // namespace

TEST(ATLASTruthSelection, RetainsForwardEnergyCandidatesAndInclusivePtBoundary)
{
  using namespace copcore::units;
  // The forward photons have pT = 0: the old pT-only policy loses them even when
  // their energy passes Athena's CaloDecay child cut. Include both threshold edges,
  // massive-particle momentum, the parent gate and the diagnostic return override.
  const std::array<Candidate, 10> candidates{{
      {500. * MeV, {0., 0., 1.}, 0., true, false, true},
      {1500. * MeV, {0., 0., 1.}, 0., true, false, true},
      {499.999 * MeV, {0., 0., 1.}, 0., true, false, false},
      {300. * MeV, {1., 0., 0.}, 0., true, false, true},
      {299.999 * MeV, {1., 0., 0.}, 0., true, false, false},
      {299.5 * MeV, {1., 0., 0.}, copcore::units::kElectronMassC2, true, false, true},
      {1. * MeV, {1., 0., 0.}, copcore::units::kElectronMassC2, true, false, false},
      {1500. * MeV, {0., 0., 1.}, 0., false, false, false},
      {1. * MeV, {0., 0., 1.}, 0., false, true, true},
      {500. * MeV, {0., 0., 1.}, copcore::units::kElectronMassC2, true, false, true},
  }};
  Candidate *deviceCandidates = nullptr;
  bool *selected              = nullptr;
  ASSERT_EQ(adept_test::MallocDeviceVisible(&deviceCandidates, sizeof(candidates)), ADEPT_DEVICE_API_SYMBOL(Success));
  ASSERT_EQ(adept_test::MallocDeviceVisible(&selected, candidates.size() * sizeof(bool)),
            ADEPT_DEVICE_API_SYMBOL(Success));
  std::copy(candidates.begin(), candidates.end(), deviceCandidates);
  SelectCandidates<<<1, 32>>>(deviceCandidates, selected, candidates.size());
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)(), ADEPT_DEVICE_API_SYMBOL(Success));
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    EXPECT_EQ(selected[i], candidates[i].expected) << "candidate " << i;
  }
  EXPECT_EQ(ADEPT_DEVICE_API_SYMBOL(Free)(selected), ADEPT_DEVICE_API_SYMBOL(Success));
  EXPECT_EQ(ADEPT_DEVICE_API_SYMBOL(Free)(deviceCandidates), ADEPT_DEVICE_API_SYMBOL(Success));
}
