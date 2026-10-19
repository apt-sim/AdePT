// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <AdePT/transport/kernels/AdePTSteppingActionSelector.cuh>
#include <AdePT/transport/kernels/WoodcockHelper.cuh>
#include <AdePT/transport/queues/ParticleManager.cuh>
#include <AdePT/transport/steps/GPUStep.hh>
#include <AdePT/transport/tracks/ParticleTypes.hh>

#include <G4HepEmConstants.hh>

namespace adept::transport {

/// @brief Annihilate a stopped positron into two back-to-back gammas (isotropic), shared by the monolithic and the
/// split kernels.
/// @details The stepping action classifies each gamma separately, so it can reject either of them; In the current
/// implementation, a rejected gamma is dropped without depositing its energy (see
/// adept::SteppingAction::SecondaryDecision). This is correct for gammas that are cut due to Russian Roulette but would
/// be incorrect for most other cuts. However, this is currently mimicking the behavior of the stepping action in the
/// LHC experiments.
/// @tparam AdvanceRNGBeforeBranch Before branching the first gamma's RNG state, the split kernels advance the parent
/// state while the monolithic kernel only branches it; both are kept to preserve the established RNG sequences.
/// @param pos,navState,globalTime kinematics at the annihilation point: the monolithic kernel passes its step-local
/// state, the split kernels pass the track members
template <class SteppingActionT, bool AdvanceRNGBeforeBranch>
__device__ __forceinline__ void PerformStoppedAnnihilation(
    ChargedTrack &currentTrack, ParticleManager &particleManager, double &energyDeposit, const bool ApplyCuts,
    const double theGammaCut, vecgeom::Vector3D<double> const &pos, vecgeom::NavigationState const &navState,
    const double globalTime, SecondaryInitData *secondaryData, unsigned int &nSecondaries,
    adeptint::VolAuxData const &auxData, typename SteppingActionT::Params const &params, const bool returnLastStep)
{
  // Apply cuts
  if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut)) {
    // Deposit the energy here and don't initialize any secondaries
    energyDeposit += 2 * copcore::units::kElectronMassC2;
    return;
  }

  const double cost = 2 * currentTrack.Uniform() - 1;
  const double sint = sqrt(1 - cost * cost);
  const double phi  = k2Pi * currentTrack.Uniform();
  double sinPhi, cosPhi;
  sincos(phi, &sinPhi, &cosPhi);

  // As the RNG state branched for the interaction may have already been used, a fresh one is branched here
  if constexpr (AdvanceRNGBeforeBranch) {
    currentTrack.rngState.Advance();
  }
  RanluxppDouble newRNG(currentTrack.rngState.Branch());

  const bool useWDT      = ShouldUseWDT(navState, double{copcore::units::kElectronMassC2});
  auto &gammaPartManager = useWDT ? particleManager.gammasWDT : particleManager.gammas;
  const vecgeom::Vector3D<double> gamma1Dir{sint * cosPhi, sint * sinPhi, cost};

  // The gammas are created one by one, as the stepping action may reject either of them; a rejected gamma
  // is dropped without depositing its energy (see SecondaryDecision)
  for (int iGamma = 0; iGamma < 2; ++iGamma) {
    const auto decision = SteppingActionT::ClassifySecondary(
        currentTrack, ParticleType::Gamma, copcore::units::kElectronMassC2, pos, globalTime, auxData, params);
    if (!decision) continue;

    // The second gamma reuses the RNG state of the dying track.
    auto &gamma = gammaPartManager.NextTrack(
        iGamma == 0 ? newRNG : currentTrack.rngState, double{copcore::units::kElectronMassC2}, pos,
        iGamma == 0 ? gamma1Dir : -gamma1Dir, navState, currentTrack, globalTime, decision.weight);
    adept::SteppingAction::SetSecondaryHostData<SteppingActionT>(gamma, currentTrack, 0., returnLastStep);

    // Return the initializing step when HostTrackData is needed for this secondary.
    if (gamma.hasHostData) {
      secondaryData[nSecondaries++] = {gamma.trackId,
                                       gamma.dir,
                                       gamma.eKin,
                                       gamma.weight,
                                       /*creator process: annihilation*/ short(2),
                                       ParticleType::Gamma,
                                       gamma.hasHostData};
    }
  }
}

} // namespace adept::transport
