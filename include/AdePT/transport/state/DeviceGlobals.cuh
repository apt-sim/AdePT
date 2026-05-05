// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_STATE_DEVICE_GLOBALS_CUH
#define ADEPT_TRANSPORT_STATE_DEVICE_GLOBALS_CUH

#include <AdePT/transport/geometry/GeometryAuxData.hh>
#include <AdePT/transport/magneticfield/GeneralMagneticField.cuh>
#include <AdePT/transport/magneticfield/UniformMagneticField.cuh>
#include <AdePT/transport/support/SystemOfUnits.h>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>

namespace AsyncAdePT {

// Constant data structures from G4HepEm accessed by the kernels.
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

// Pointer for array of volume auxiliary data on device.
extern __constant__ __device__ adeptint::VolAuxData *gVolAuxData;

extern __constant__ __device__ adeptint::WDTDeviceView gWDTData;

constexpr double kPush = 1.e-8 * copcore::units::cm;

#ifdef ADEPT_USE_EXT_BFIELD
extern __constant__ __device__ GeneralMagneticField *gMagneticField;
#else
extern __constant__ __device__ UniformMagneticField *gMagneticField;
#endif

} // namespace AsyncAdePT

#endif
