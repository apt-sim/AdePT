// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <AdePT/core/GeometryAuxData.hh>
#include <AdePT/copcore/Ranluxpp.h>
#include <AdePT/copcore/SystemOfUnits.h>

#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialData.hh>

#include <VecGeom/base/Vector3D.h>

#include <cmath>

namespace adept::SteppingAction {

struct GammaRouletteResult {
  bool create{true};  ///< whether the gamma secondary should be created
  float weight{1.0f}; ///< weight assigned to the newborn gamma when created
};

///< @brief helper function to kill a track and deposit its energy
__device__ __forceinline__ void KillTrack(bool &alive, double &eKin, double &edep)
{
  if (!alive) return;
  // NOTE: currently, positrons are killed directly. If done correctly,
  // the energy of 2 * m_e would be additionally deposited to account for the annihilation at rest.
  // This is omitted here to match what CMS and LHCb are doing.
  alive = false;
  edep += eKin;
  eKin = 0.0;
}

// ---------- No-op (NONE) ----------
// default: no AdePT-specific SteppingAction
struct NoAction {
  // empty placeholder struct
  struct Params {};
  static constexpr bool kGammaRussianRoulette = false;

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

  __device__ __forceinline__ static GammaRouletteResult ApplyGammaRussianRoulette(float const &parentWeight,
                                                                                  double const &,
                                                                                  adeptint::VolAuxData const &,
                                                                                  RanluxppDouble &)
  {
    return {true, parentWeight};
  }
};

// ---------- CMS ----------
// Custom SteppingAction for CMS
struct CMSAction {
  struct Params {
    double tmax    = 2000.0 * copcore::units::nanosecond;               // time cut
    double zmax    = 50.0 * copcore::units::meter;                      // longitudinal envelope
    double ecut    = 2.0 * copcore::units::MeV;                         //  kinetic E for the vacuum cut
    double density = 1e-15 * (copcore::units::g / copcore::units::cm3); // density for the vacuum cut
  };
  static constexpr bool kGammaRussianRoulette = false;

  __device__ __forceinline__ static bool IsDeadRegion(adeptint::VolAuxData const &auxData)
  {
#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 1)
    return auxData.fCMSDeadRegion;
#else
    return false;
#endif
  }

  __device__ __forceinline__ static void AllParticleCheck(bool &alive, double &eKin, double &edep,
                                                          vecgeom::Vector3D<double> const &pos,
                                                          double const &globalTime, adeptint::VolAuxData const &auxData,
                                                          Params const &params)
  {
    // Configured CMSSW dead regions from AdePTConfiguration.
    if (IsDeadRegion(auxData)) {
      KillTrack(alive, eKin, edep);
      return;
    }

    // Out-of-time and out-of-z cut
    if (globalTime > params.tmax && fabs(pos.z()) >= params.zmax) {
      KillTrack(alive, eKin, edep);
      return;
    }
  }

  __device__ __forceinline__ static void ElectronAction(bool &alive, double &eKin, double &edep,
                                                        vecgeom::Vector3D<double> const &pos, double const &globalTime,
                                                        adeptint::VolAuxData const &auxData,
                                                        G4HepEmData const *g4HepEmData, Params const &params)
  {
    if (!alive) return;
    AllParticleCheck(alive, eKin, edep, pos, globalTime, auxData, params);
    if (!alive) return;

    // e-/e+ in vacuum cut
    const int hmi        = g4HepEmData->fTheMatCutData->fMatCutData[auxData.fMCIndex].fHepEmMatIndex;
    const double density = g4HepEmData->fTheMaterialData->fMaterialData[hmi].fDensity; // g/cm^3
    if (eKin < params.ecut && density < params.density) {
      KillTrack(alive, eKin, edep);
    }
  }

  __device__ __forceinline__ static void GammaAction(bool &alive, double &eKin, double &edep,
                                                     vecgeom::Vector3D<double> const &pos, double const &globalTime,
                                                     adeptint::VolAuxData const &auxData,
                                                     G4HepEmData const * /*g4HepEmData*/, Params const &params)
  {
    if (!alive) return;
    AllParticleCheck(alive, eKin, edep, pos, globalTime, auxData, params);
  }

  __device__ __forceinline__ static GammaRouletteResult ApplyGammaRussianRoulette(float const &parentWeight,
                                                                                  double const &,
                                                                                  adeptint::VolAuxData const &,
                                                                                  RanluxppDouble &)
  {
    return {true, parentWeight};
  }
};

// ---------- LHCb ----------
// Custom SteppingAction for LHCb
struct LHCbAction {
  struct Params {
    // default parameters for world cut
    double xmin{-10000 * copcore::units::millimeter}, xmax{10000 * copcore::units::millimeter},
        ymin{-10000 * copcore::units::millimeter}, ymax{10000 * copcore::units::millimeter},
        zmin{-5000 * copcore::units::millimeter}, zmax{25000 * copcore::units::millimeter};
    // default parameters for energy tracking cut
    double ecut = 1.0 * copcore::units::MeV;
  };
  static constexpr bool kGammaRussianRoulette = false;

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

  __device__ __forceinline__ static GammaRouletteResult ApplyGammaRussianRoulette(float const &parentWeight,
                                                                                  double const &,
                                                                                  adeptint::VolAuxData const &,
                                                                                  RanluxppDouble &)
  {
    return {true, parentWeight};
  }
};

// ---------- ATLAS ----------
// The ATLAS SteppingAction must be guarded as it requires a compile time specific
// paramater in the VolumeAuxData
#if defined(ADEPT_STEPACTION_TYPE) && (ADEPT_STEPACTION_TYPE == 3)
// Custom SteppingAction for ATLAS.
// Photon Russian Roulette follows Athena's stacking-action policy for photons
// born in LAr volumes, adapted to act before the secondary gamma is created on
// the GPU. The current threshold and weight are intentionally hardcoded here to
// match the Athena defaults until AdePT has an experiment-facing configuration
// hook for these values.
struct ATLASAction {
  struct Params {};
  static constexpr bool kGammaRussianRoulette = true;

  static constexpr double kPhotonRussianRouletteThreshold = 0.5 * copcore::units::MeV;
  static constexpr float kPhotonRussianRouletteWeight     = 10.0f;
  static constexpr float kOneOverPhotonRouletteWeight     = 1.0f / kPhotonRussianRouletteWeight;

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

  __device__ __forceinline__ static GammaRouletteResult ApplyGammaRussianRoulette(
      float const &parentWeight, double const &gammaEkin, adeptint::VolAuxData const &originAuxData,
      RanluxppDouble &rngState)
  {
    if (!originAuxData.fAtlasPhotonRussianRoulette) return {true, parentWeight};
    if (parentWeight >= kPhotonRussianRouletteWeight) return {true, parentWeight};
    if (gammaEkin >= kPhotonRussianRouletteThreshold) return {true, parentWeight};

    if (rngState.Rndm() > kOneOverPhotonRouletteWeight) return {false, parentWeight};

    return {true, kPhotonRussianRouletteWeight};
  }
};
#else
// Stub used in non-ATLAS builds so the selector can still name ATLASAction
// without pulling in any ATLAS-only aux-data fields or runtime logic.
struct ATLASAction {
  struct Params {};
  static constexpr bool kGammaRussianRoulette = false;

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

  __device__ __forceinline__ static GammaRouletteResult ApplyGammaRussianRoulette(float const &parentWeight,
                                                                                  double const &,
                                                                                  adeptint::VolAuxData const &,
                                                                                  RanluxppDouble &)
  {
    return {true, parentWeight};
  }
};
#endif

} // namespace adept::SteppingAction
