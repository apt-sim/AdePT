// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <AdePT/core/TrackData.h>
#include <AdePT/copcore/SystemOfUnits.h>

#include <G4HepEmData.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialData.hh>

#include <VecGeom/base/Vector3D.h>

#include <cmath>

namespace adept::SteppingAction {

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

  __device__ __forceinline__ static void ElectronAction(bool &, double &, double &, vecgeom::Vector3D<double> const &,
                                                        double const &, int const &, G4HepEmData const *,
                                                        Params const &)
  {
  }

  __device__ __forceinline__ static void GammaAction(bool &, double &, double &, vecgeom::Vector3D<double> const &,
                                                     double const &, int const &, G4HepEmData const *, Params const &)
  {
  }
};

// ---------- CMS ----------
// Custom SteppingAction for Atlas
struct CMSAction {
  struct Params {
    double tmax    = 2000.0 * copcore::units::nanosecond;               // time cut
    double zmax    = 50.0 * copcore::units::meter;                      // longitudinal envelope
    double ecut    = 2.0 * copcore::units::MeV;                         //  kinetic E for the vacuum cut
    double density = 1e-15 * (copcore::units::g / copcore::units::cm3); // density for the vacuum cut
  };

  __device__ __forceinline__ static void AllParticleCheck(bool &alive, double &eKin, double &edep,
                                                          vecgeom::Vector3D<double> const &pos,
                                                          double const &globalTime, Params const &params)
  {
    // dead-region cut:
    // Missing: mark dead material regions in CMS
    // Before, it was done via the LeakStatus, but it is not clear whether the parametrized shower will ever be used on
    // the GPU, so the LeakStatus::OutOfGPURegion is a bad marker. Just left here for documenting purposes. Instead, the
    // dead regions must be implemented for CMS, to check with the NavigationState if (leak ==
    // LeakStatus::OutOfGPURegion) {
    //   KillTrack(alive, eKin, edep, leak);
    //   return;
    // }

    // Out-of-time and out-of-z cut
    if (globalTime > params.tmax && fabs(pos.z()) >= params.zmax) {
      KillTrack(alive, eKin, edep);
      return;
    }
  }

  __device__ __forceinline__ static void ElectronAction(bool &alive, double &eKin, double &edep,
                                                        vecgeom::Vector3D<double> const &pos, double const &globalTime,
                                                        int const &mcIndex, G4HepEmData const *g4HepEmData,
                                                        Params const &params)
  {
    if (!alive) return;
    AllParticleCheck(alive, eKin, edep, pos, globalTime, params);
    if (!alive) return;

    // e-/e+ in vacuum cut
    const int hmi        = g4HepEmData->fTheMatCutData->fMatCutData[mcIndex].fHepEmMatIndex;
    const double density = g4HepEmData->fTheMaterialData->fMaterialData[hmi].fDensity; // g/cm^3
    if (eKin < params.ecut && density < params.density) {
      KillTrack(alive, eKin, edep);
    }
  }

  __device__ __forceinline__ static void GammaAction(bool &alive, double &eKin, double &edep,
                                                     vecgeom::Vector3D<double> const &pos, double const &globalTime,
                                                     int const & /*mcIndex*/, G4HepEmData const * /*g4HepEmData*/,
                                                     Params const &params)
  {
    if (!alive) return;
    AllParticleCheck(alive, eKin, edep, pos, globalTime, params);
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

  __device__ __forceinline__ static void ElectronAction(bool &alive, double &eKin, double &edep,
                                                        vecgeom::Vector3D<double> const &pos,
                                                        double const & /*globalTime*/, int const & /*mcIndex*/,
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
                                                     double const & /*globalTime*/, int const & /*mcIndex*/,
                                                     G4HepEmData const *g4HepEmData, Params const &params)
  {
    // same as electron stepping action
    ElectronAction(alive, eKin, edep, pos, /*globalTime*/ 0.0, /*mcIndex*/ 0, g4HepEmData, params);
  }
};

} // namespace adept::SteppingAction
