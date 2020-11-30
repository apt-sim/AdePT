// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief   Physical constants in internal units.
 * @file    CopCore/PhysicalConstants.h
 * @author  M Novak, A Ribon
 * @date    december 2015
 *
 * This is porting the corresponding file from gitlab.cern.ch/GeantV, adopting CopCore namespace.
 * Original file was taken from CLHEP and changed according to GeantV.
 * Authors of the original version: M. Maire
 *
 * The basic units are :
 * \li  centimeter              (centimeter)
 * \li  second                  (second)
 * \li  giga electron volt      (GeV)
 * \li  positron charge         (eplus)
 * \li  degree Kelvin           (kelvin)
 * \li  the amount of substance (mole)
 * \li  luminous intensity      (candela)
 * \li  radian                  (radian)
 * \li  steradian               (steradian)
 *
 * You can add your own constants.
 *
 */

#ifndef COPCORE_PHYSICALCONSTANTS_H
#define COPCORE_PHYSICALCONSTANTS_H

#include <CopCore/SystemOfUnits.h>

namespace copcore {
namespace units {

//
//
//
static constexpr double kAvogadro = 6.02214179e+23 / mole;

//
// kCLight       ->   c   = 2.9979246e+10 [cm/s]
// kCLightSquare ->   c^2 = 8.9875518e+20 [(cm/s)^2]
//
static constexpr double kCLight       = 2.99792458e+8 * m / s;
static constexpr double kCLightSquare = kCLight * kCLight;

//
// kHPlanck          -> h        = 4.1356673e-24 [GeV*s]
// kHBarPlanck       -> h/(2Pi)  = 6.582119e-25  [GeV*s]
// kHBarPlanckCLight -> hc/(2Pi) = 1.9732696e-14 [GeV*cm]
//
static constexpr double kHPlanck                = 6.62606896e-34 * joule * s;
static constexpr double kHBarPlanck             = kHPlanck / kTwoPi;
static constexpr double kHBarPlanckCLight       = kHBarPlanck * kCLight;
static constexpr double kHBarPlanckCLightSquare = kHBarPlanckCLight * kHBarPlanckCLight;

//
//
//
static constexpr double kElectronCharge   = -eplus; // see Geant/SystemOfUnits.h
static constexpr double kUnitChargeSquare = eplus * eplus;

//
// kAtomicMassUnitC2 -> atomic equivalent mass unit
//                      AKA, unified atomic mass unit (u)
// kAtomicMassUnit   -> atomic mass unit
//
static constexpr double kElectronMassC2    = 0.510998910 * MeV;
static constexpr double kInvElectronMassC2 = 1.0 / kElectronMassC2;
static constexpr double kProtonMassC2      = 938.272013 * MeV;
static constexpr double kNeutronMassC2     = 939.56536 * MeV;
static constexpr double kAtomicMassUnitC2  = 931.494028 * MeV;
static constexpr double kAtomicMassUnit    = kAtomicMassUnitC2 / kCLightSquare;

//
// kMu0      -> permeability of free space  = 2.0133544e-36 [GeV*(s*eplus)^2/cm]
// kEpsilon0 -> permittivity of free space  = 5.5263499e+14 [eplus^2/(GeV*cm)]
//
static constexpr double kMu0      = 4 * kPi * 1.e-7 * henry / m;
static constexpr double kEpsilon0 = 1. / (kCLightSquare * kMu0);

//
// kEMCoupling            -> electromagnetic coupling  = 1.4399644e-16 [GeV*cm/(eplus^2)]
// kFineStructConst       -> fine structure constant   = 0.0072973525 []
// kClassicElectronRadius -> classical electron radius = 2.8179403e-13 [cm]
// kRedElectronComptonWLenght -> reduced electron Compton wave length = 3.8615926e-10 [cm]
// kBohrRadius            -> Bohr radius               = 5.2917721e-08 [cm]
static constexpr double kEMCoupling                = kUnitChargeSquare / (4 * kPi * kEpsilon0);
static constexpr double kFineStructConst           = kEMCoupling / kHBarPlanckCLight;
static constexpr double kClassicElectronRadius     = kEMCoupling / kElectronMassC2;
static constexpr double kRedElectronComptonWLenght = kHBarPlanckCLight / kElectronMassC2;
static constexpr double kBohrRadius                = kRedElectronComptonWLenght / kFineStructConst;

//
//
//
static constexpr double kBoltzmann = 8.617343e-11 * MeV / kelvin;

//
// STP -> Standard Temperature and Pressure
// NTP -> Normal Temperature and Pressure
//
static constexpr double kSTPTemperature = 273.15 * kelvin;
static constexpr double kNTPTemperature = 293.15 * kelvin;
static constexpr double kSTPPressure    = 1. * atmosphere;
static constexpr double kGasThreshold   = 10. * mg / cm3;

//
//
//
static constexpr double kUniverseMeanDensity = 1.e-25 * g / cm3;

} // namespace units
} // namespace copcore

#endif // COPCORE_PHYSICALCONSTANTS_H
