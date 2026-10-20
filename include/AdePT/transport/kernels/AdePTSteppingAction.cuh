// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <AdePT/AdePTBuildFlags.hh>

#include <AdePT/transport/geometry/GeometryAuxData.hh>
#include <AdePT/transport/random/Ranluxpp.h>
#include <AdePT/transport/config/SteppingActionParams.hh>
#include <AdePT/transport/support/SystemOfUnits.h>
#include <AdePT/transport/tracks/ParticleTypes.hh>
#include <AdePT/transport/tracks/Track.cuh>

#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialData.hh>

#include <VecGeom/base/Vector3D.h>

#include <cmath>

namespace adept::SteppingAction {

/// @brief Decision on a secondary that is about to be created.
/// @details Evaluated before the secondary track is created. In contrast to a secondary below the production cut,
/// a rejected secondary is dropped WITHOUT depositing its energy: this mirrors the experiments' track filters
/// (e.g. the CMSSW stacking filter), which discard filtered secondaries without a local energy deposit.
struct SecondaryDecision {
  bool create{true};  ///< whether the secondary is created
  float weight{1.0f}; ///< weight of the secondary when created

  __host__ __device__ constexpr explicit operator bool() const { return create; }
};

///< @brief helper function to kill a track and deposit its energy. Currently only used by LHCb SteppingAction
__device__ __forceinline__ void KillTrack(bool &alive, double &eKin, double &edep)
{
  if (!alive) return;
  // NOTE: currently, positrons are killed directly. If done correctly,
  // the energy of 2 * m_e would be additionally deposited to account for the annihilation at rest.
  // This is omitted here to match what LHCb is doing.
  alive = false;
  edep += eKin;
  eKin = 0.0;
}

// ---------- No-op (NONE) ----------
// default: no AdePT-specific SteppingAction
struct NoAction {
  using Params = EmptySteppingActionParams;

  __device__ __forceinline__ static void ElectronAction(bool &, double &, double &, vecgeom::Vector3D<double> const &,
                                                        double const &, adeptint::VolAuxData const &,
                                                        G4HepEmData const *, Params const &)
  {
  }

  __device__ __forceinline__ static void GammaAction(bool &, double &, double &, vecgeom::Vector3D<double> const &,
                                                     double const &, adeptint::VolAuxData const &, G4HepEmData const *,
                                                     Params const &)
  {
  }

  __device__ __forceinline__ static SecondaryDecision ClassifySecondary(TrackBase &parentTrack, ParticleType, double,
                                                                        vecgeom::Vector3D<double> const &, double,
                                                                        adeptint::VolAuxData const &, Params const &)
  {
    return {true, parentTrack.weight};
  }

  __device__ __forceinline__ static bool NeedsHostTrackData(double, vecgeom::Vector3D<double> const &, double)
  {
    return false;
  }
};

// ---------- CMS ----------
// Custom SteppingAction for CMS (Phase 2). It mirrors the CMSSW Phase2SteppingAction, which kills tracks after a step,
// and its Phase2TrackFilter, which kills new secondaries or plays Russian roulette with them. When CMSSW receives the
// replayed GPU steps, the filter erases (and deletes) the secondaries that it kills, although they are already
// transported on the GPU, so it must be applied before a secondary is created.
// Then, no particle returned from the GPU must still be killed by CMSSW itself.
// For Russian Roulette that is guaranteed, as surviving photons have the increased weight, preventing the
// re-rouletting. The cuts are configured by CMSSW, see CMSSteppingActionConfig.
struct CMSAction {
  using Params = CMSSteppingActionParams;

  // The volume flags below only exist in VolAuxData for CMS builds
  __device__ __forceinline__ static bool IsDeadRegion(adeptint::VolAuxData const &auxData)
  {
#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 1)
    return auxData.fCMSDeadRegion;
#else
    return false;
#endif
  }

  __device__ __forceinline__ static bool IsLowDensityRegion(adeptint::VolAuxData const &auxData)
  {
#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 1)
    return auxData.fCMSLowDensityRegion;
#else
    return false;
#endif
  }

  /// @brief Survival probability of the gamma Russian roulette for a gamma born in this volume (1: no roulette)
  __device__ __forceinline__ static double GammaRussianRouletteProbability(adeptint::VolAuxData const &auxData,
                                                                           bool parentApplicable)
  {
#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 1)
    if (auxData.fCMSGammaRussianRouletteRestricted && !parentApplicable) return 1.;
    return auxData.fCMSGammaRussianRouletteProbability;
#else
    return 1.;
#endif
  }

  /// @brief Kill a track like Phase2SteppingAction, which does not deposit the remaining energy.
  __device__ __forceinline__ static void Kill(bool &alive, double &eKin)
  {
    alive = false;
    eKin  = 0.;
  }

  /// @brief Kills of Phase2SteppingAction that apply to all particles.
  /// @return whether the track is in the central region, where CMSSW applies its other cuts as well
  __device__ __forceinline__ static bool AllParticleCheck(bool &alive, double &eKin,
                                                          vecgeom::Vector3D<double> const &pos,
                                                          double const &globalTime, adeptint::VolAuxData const &auxData,
                                                          Params const &params)
  {
    if (fabs(pos.z()) >= params.maxZCentral) {
      // Very forward region: only the forward time cut applies
      if (globalTime > params.maxTrackTimeForward) Kill(alive, eKin);
      return false;
    }
    if (IsDeadRegion(auxData) || globalTime > params.maxTrackTime) Kill(alive, eKin);
    return true;
  }

  __device__ __forceinline__ static void ElectronAction(bool &alive, double &eKin, double & /*edep*/,
                                                        vecgeom::Vector3D<double> const &pos, double const &globalTime,
                                                        adeptint::VolAuxData const &auxData,
                                                        G4HepEmData const *g4HepEmData, Params const &params)
  {
    if (!alive) return;
    const bool central = AllParticleCheck(alive, eKin, pos, globalTime, auxData, params);
    if (!alive || !central) return;

    // e-/e+ in vacuum cut
    const int hmi        = g4HepEmData->fTheMatCutData->fMatCutData[auxData.fMCIndex].fHepEmMatIndex;
    const double density = g4HepEmData->fTheMaterialData->fMaterialData[hmi].fDensity; // g/cm^3
    if (eKin < params.criticalEnergyForVacuum && density <= params.criticalDensity) Kill(alive, eKin);
  }

  __device__ __forceinline__ static void GammaAction(bool &alive, double &eKin, double & /*edep*/,
                                                     vecgeom::Vector3D<double> const &pos, double const &globalTime,
                                                     adeptint::VolAuxData const &auxData,
                                                     G4HepEmData const * /*g4HepEmData*/, Params const &params)
  {
    if (!alive) return;
    AllParticleCheck(alive, eKin, pos, globalTime, auxData, params);
  }

  /// @brief Decision of Phase2TrackFilter::ClassifyNewTrack on a secondary, taken before it is created.
  /// @details A killed secondary is dropped without depositing its energy, as CMSSW adds the energy of filtered
  /// secondaries to the step only after the sensitive detectors have processed it. This is the intrinsic Geant4 calling
  /// order.
  //  As in CMSSW, Russian roulette only applies to gammas with a weight of at most one. Survivors get a weight above
  //  one if the parent weight is above
  /// the survival probability, which holds in CMS, so that CMSSW does not play Russian roulette with them again.
  __device__ __forceinline__ static SecondaryDecision ClassifySecondary(
      TrackBase &parentTrack, ParticleType type, double eKin, vecgeom::Vector3D<double> const &pos, double globalTime,
      adeptint::VolAuxData const &auxData, Params const &params)
  {
    if ((fabs(pos.z()) >= params.maxZCentral && globalTime > params.maxTrackTimeForward) ||
        globalTime > params.maxTrackTime || IsDeadRegion(auxData) ||
        (eKin <= params.criticalEnergyForVacuum && IsLowDensityRegion(auxData)) ||
        (type == ParticleType::Gamma && eKin < params.gammaEnergyCut)) {
      return {false, parentTrack.weight};
    }

    const double probability = GammaRussianRouletteProbability(auxData, parentTrack.gammaRussianRouletteApplicable);
    if (type != ParticleType::Gamma || probability >= 1. || parentTrack.weight > 1.f ||
        eKin >= params.gammaRussianRouletteEnergyLimit) {
      return {true, parentTrack.weight};
    }
    if (parentTrack.rngState.Rndm() >= probability) return {false, parentTrack.weight};
    return {true, static_cast<float>(parentTrack.weight / probability)};
  }

  __device__ __forceinline__ static bool NeedsHostTrackData(double, vecgeom::Vector3D<double> const &, double)
  {
    return false;
  }
};

// ---------- LHCb ----------
// Custom SteppingAction for LHCb
struct LHCbAction {
  using Params = LHCbSteppingActionParams;

  __device__ __forceinline__ static void ElectronAction(bool &alive, double &eKin, double &edep,
                                                        vecgeom::Vector3D<double> const &pos,
                                                        double const & /*globalTime*/,
                                                        adeptint::VolAuxData const & /*auxData*/,
                                                        G4HepEmData const * /*g4HepEmData*/, Params const &params)
  {
    if (!alive) return;

    // out-of-world and energy tracking cut
    if (pos.x() > params.xmax || pos.x() < params.xmin || pos.y() > params.ymax || pos.y() < params.ymin ||
        pos.z() > params.zmax || pos.z() < params.zmin || eKin < params.ecut) {
      KillTrack(alive, eKin, edep);
    }
  }

  __device__ __forceinline__ static void GammaAction(bool &alive, double &eKin, double &edep,
                                                     vecgeom::Vector3D<double> const &pos,
                                                     double const & /*globalTime*/, adeptint::VolAuxData const &auxData,
                                                     G4HepEmData const *g4HepEmData, Params const &params)
  {
    // same as electron stepping action
    ElectronAction(alive, eKin, edep, pos, /*globalTime*/ 0.0, auxData, g4HepEmData, params);
  }

  __device__ __forceinline__ static SecondaryDecision ClassifySecondary(TrackBase &parentTrack, ParticleType, double,
                                                                        vecgeom::Vector3D<double> const &, double,
                                                                        adeptint::VolAuxData const &, Params const &)
  {
    return {true, parentTrack.weight};
  }

  __device__ __forceinline__ static bool NeedsHostTrackData(double, vecgeom::Vector3D<double> const &, double)
  {
    return false;
  }
};

// ---------- ATLAS ----------
// The ATLAS SteppingAction must be guarded as it requires a compile time specific
// parameter in the VolumeAuxData
#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 3)
// Custom SteppingAction for ATLAS.
// Photon Russian Roulette follows Athena's stacking-action policy for photons
// born in LAr volumes, adapted to act before the secondary gamma is created on
// the GPU. The current threshold and weight are intentionally hardcoded here to
// match the Athena defaults until AdePT has an experiment-facing configuration
// hook for these values.
struct ATLASAction {
  using Params = EmptySteppingActionParams;

  static constexpr double kPhotonRussianRouletteThreshold = 0.5 * copcore::units::MeV;
  static constexpr float kPhotonRussianRouletteWeight     = 10.0f;
  static constexpr float kOneOverPhotonRouletteWeight     = 1.0f / kPhotonRussianRouletteWeight;
  static constexpr double kTruthMinPt                     = 300. * copcore::units::MeV;

  __device__ __forceinline__ static bool NeedsHostTrackData(double eKin, vecgeom::Vector3D<double> const &direction,
                                                            double mass)
  {
    const double p2    = mass > 0. ? eKin * (eKin + 2. * mass) : eKin * eKin;
    const double dirT2 = direction.x() * direction.x() + direction.y() * direction.y();
    return p2 * dirT2 > kTruthMinPt * kTruthMinPt;
  }

  __device__ __forceinline__ static void ElectronAction(bool &, double &, double &, vecgeom::Vector3D<double> const &,
                                                        double const &, adeptint::VolAuxData const &,
                                                        G4HepEmData const *, Params const &)
  {
  }

  __device__ __forceinline__ static void GammaAction(bool &, double &, double &, vecgeom::Vector3D<double> const &,
                                                     double const &, adeptint::VolAuxData const &, G4HepEmData const *,
                                                     Params const &)
  {
  }

  __device__ __forceinline__ static SecondaryDecision ClassifySecondary(TrackBase &parentTrack, ParticleType type,
                                                                        double eKin, vecgeom::Vector3D<double> const &,
                                                                        double,
                                                                        adeptint::VolAuxData const &originAuxData,
                                                                        Params const &)
  {
    if (type != ParticleType::Gamma || !originAuxData.fAtlasPhotonRussianRoulette) return {true, parentTrack.weight};
    if (parentTrack.weight != 1.0f) return {true, parentTrack.weight};
    if (eKin >= kPhotonRussianRouletteThreshold) return {true, parentTrack.weight};

    if (parentTrack.rngState.Rndm() > kOneOverPhotonRouletteWeight) return {false, parentTrack.weight};

    return {true, kPhotonRussianRouletteWeight};
  }
};
#else
// Stub used in non-ATLAS builds so the selector can still name ATLASAction
// without pulling in any ATLAS-only aux-data fields or runtime logic.
struct ATLASAction {
  using Params = EmptySteppingActionParams;

  __device__ __forceinline__ static void ElectronAction(bool &, double &, double &, vecgeom::Vector3D<double> const &,
                                                        double const &, adeptint::VolAuxData const &,
                                                        G4HepEmData const *, Params const &)
  {
  }

  __device__ __forceinline__ static void GammaAction(bool &, double &, double &, vecgeom::Vector3D<double> const &,
                                                     double const &, adeptint::VolAuxData const &, G4HepEmData const *,
                                                     Params const &)
  {
  }

  __device__ __forceinline__ static SecondaryDecision ClassifySecondary(TrackBase &parentTrack, ParticleType, double,
                                                                        vecgeom::Vector3D<double> const &, double,
                                                                        adeptint::VolAuxData const &, Params const &)
  {
    return {true, parentTrack.weight};
  }

  __device__ __forceinline__ static bool NeedsHostTrackData(double, vecgeom::Vector3D<double> const &, double)
  {
    return false;
  }
};
#endif

template <class ActionT>
__device__ __forceinline__ void SetSecondaryHostData(TrackBase &child, TrackBase const &parent, double childMass,
                                                     bool returnLastStep)
{
  child.hasHostData =
      returnLastStep || (parent.hasHostData && ActionT::NeedsHostTrackData(child.eKin, child.dir, childMass));
}

} // namespace adept::SteppingAction
