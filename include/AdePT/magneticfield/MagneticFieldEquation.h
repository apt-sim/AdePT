// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  J. Apostolakis,  12-17 Nov 2021
//
//  Equation of motion for pure magnetic field for
//  use in solving ODEs of motion using Runge-Kutta methods.
//

#ifndef MagneticFieldEquation_H_
#define MagneticFieldEquation_H_

#include <AdePT/copcore/PhysicalConstants.h>

// #include <VecCore/Math.h>
#include <VecGeom/base/Vector3D.h>

#include <iostream> // For cout only

template <typename MagneticField_t>
class MagneticFieldEquation {
public:
  template <typename Real_t>
  static inline __host__ __device__ void EvaluateDerivatives(MagneticField_t const &magField, const Real_t y[],
                                                             int charge, Real_t dy_ds[])
  {
    // Inline evaluation to avoid redundant calls
    const vecgeom::Vector3D<Real_t> Bvec = magField.Evaluate(y[0], y[1], y[2]);

    // Directly compute the RHS
    Real_t momentum_mag     = sqrt(y[3] * y[3] + y[4] * y[4] + y[5] * y[5]);
    Real_t inv_momentum_mag = Real_t(1.0) / momentum_mag;

    dy_ds[0] = y[3] * inv_momentum_mag;
    dy_ds[1] = y[4] * inv_momentum_mag;
    dy_ds[2] = y[5] * inv_momentum_mag;

    Real_t cof = charge * gCof * inv_momentum_mag;

    dy_ds[3] = cof * (y[4] * Bvec[2] - y[5] * Bvec[1]);
    dy_ds[4] = cof * (y[5] * Bvec[0] - y[3] * Bvec[2]);
    dy_ds[5] = cof * (y[3] * Bvec[1] - y[4] * Bvec[0]);
  }

  // Key methods
  // -----------
  template <typename Real_t>
  static __host__ __device__ void EvaluateDerivativesGivenB(const Real_t y[], const Real_t Bfield[3], int charge,
                                                            Real_t dy_ds[]);
  //  Evaluate
  //       dy_ds = d/ds ( position(x, y, z),  momentum (px, py, pz) )
  //  given charge and
  //       y  = ( x, y, z, p_x, p_y, p_z )
  //   Bfield = ( B_x, B_y, B_z )

  template <typename Real_t>
  static __host__ __device__ void EvaluateDerivatives(const Real_t y[], vecgeom::Vector3D<Real_t> const &Bvec,
                                                      int charge, Real_t dy_ds[])
  {
    EvaluateRhsGivenB<Real_t, int>(y, charge, Bvec, dy_ds);
  }
  //  Same with Vector3D arguments

  // template <typename Real_t>
  // static __host__ __device__ void EvaluateDerivatives(MagneticField_t const &magField, const Real_t y[], int charge,
  //                                                     Real_t dy_ds[]);
  // Same, with MagneticField_t object, used to obtain B-field.

  template <typename Real_t>
  static __host__ __device__ void EvaluateDerivativesReturnB(MagneticField_t const &magField, const Real_t y[],
                                                             int charge, Real_t dy_ds[],
                                                             vecgeom::Vector3D<float> &BfieldVal);
  // Same, with MagneticField_t object, used to obtain B-field.

  template <typename Real_t>
  inline __host__ __device__ void EvaluateDerivativesReturnB(MagneticField_t const &magField,
                                                             vecgeom::Vector3D<float> const &position,
                                                             vecgeom::Vector3D<float> const &momentum, int charge,
                                                             Real_t dy_ds[], vecgeom::Vector3D<float> &BfieldVal);

  // Implementation method
  // ---------------------
  template <typename Real_t, typename Int_t>
  static __host__ __device__ void EvaluateRhsGivenB(const Real_t y[/*Nvar*/], const Int_t &charge,
                                                    const vecgeom::Vector3D<Real_t> &Bvalue, Real_t dy_ds[/*Nvar*/])
  {
    vecgeom::Vector3D<Real_t> momentum = {y[3], y[4], y[5]};
    Force(momentum, charge, Bvalue, dy_ds[0], dy_ds[1], dy_ds[2], dy_ds[3], dy_ds[4], dy_ds[5]);
    // Force(    y[4], y[5], B[0], B[1], B[2], charge,
    //       dy_ds[0], dy_ds[1], dy_ds[2], dy_ds[3], dy_ds[4], dy_ds[5] );
  }

  static constexpr double gCof = copcore::units::kCLight; //   / fieldUnits::meter ;
  static constexpr int Nvar    = 6;

private:
  template <typename Real_t, typename Int_t>
  static __host__ __device__ void Force( // Vector3D<Real_t> const & position,
      vecgeom::Vector3D<Real_t> const &momentum, Int_t const &charge, vecgeom::Vector3D<Real_t> const &Bfield,
      // Real_t const & momentumMag, //  Magnitude of initial momentum
      // Vector3D<Real_t>  dy_ds
      Real_t &dx_ds, Real_t &dy_ds, Real_t &dz_ds, Real_t &dpx_ds, Real_t &dpy_ds, Real_t &dpz_ds);
  /******
  template <typename Real_t, typename Int_t>
  static
  __host__ __device__ void
     Force(Real_t const & momx, Real_t const & momy, Real_t const & momz,
           Real_t const & Bx,   Real_t const & By,   Real_t const & Bz,
           Int_t  const & charge,
           // Real_t const & momentumMag, //  Magnitude of initial momentum
           Real_t &  dx_ds, Real_t &  dy_ds, Real_t &  dz_ds,
           Real_t & dpx_ds, Real_t & dpy_ds, Real_t & dpz_ds
     ) ;
  *********/
  //  Implements the Lorentz force for RK integration of d/ds ( x, y, z, px, py, pz )
};
// ------------------------------------------------------------------------------------

template <typename MagField_t>
template <typename Real_t, typename Int_t>
inline __host__ __device__ void MagneticFieldEquation<MagField_t>::Force(vecgeom::Vector3D<Real_t> const &momentum,
                                                                         Int_t const &charge,
                                                                         vecgeom::Vector3D<Real_t> const &Bfield,
                                                                         // Real_t const & momentumMag, //  Magnitude of
                                                                         // initial momentum Vector3D<Real_t>  dy_ds
                                                                         // Real_t dydsArr[/*Nvar*/]
                                                                         Real_t &dx_ds, Real_t &dy_ds, Real_t &dz_ds,
                                                                         Real_t &dpx_ds, Real_t &dpy_ds, Real_t &dpz_ds)
{
  Real_t inv_momentum_mag =
      Real_t(1.) /
      // vecCore::math::Sqrt
      std::sqrt(momentum[0] * momentum[0] + momentum[1] * momentum[1] + momentum[2] * momentum[2]);
  //    = vdt::fast_isqrt_general( momentum_sqr, 2); // Alternative

  dpx_ds = (momentum[1] * Bfield[2] - momentum[2] * Bfield[1]); //    Ax = a*(Vy*Bz - Vz*By)
  dpy_ds = (momentum[2] * Bfield[0] - momentum[0] * Bfield[2]); //    Ay = a*(Vz*Bx - Vx*Bz)
  dpz_ds = (momentum[0] * Bfield[1] - momentum[1] * Bfield[0]); //    Az = a*(Vx*By - Vy*Bx)

  Real_t cof = charge * Real_t(gCof) * inv_momentum_mag;

  dx_ds = momentum[0] * inv_momentum_mag; //  (d/ds)x = Vx/V
  dy_ds = momentum[1] * inv_momentum_mag; //  (d/ds)y = Vy/V
  dz_ds = momentum[2] * inv_momentum_mag; //  (d/ds)z = Vz/V

  dpx_ds *= cof;
  dpy_ds *= cof;
  dpz_ds *= cof;
  // std::cout << "Force(v2) - mom = " << momentum[0] << " " << momentum[1] << " " << momentum[2] << " B= " << Bfield[0]
  // << " " << Bfield[1] << " " << Bfield[2] << "\n"; std::cout << "Force(v2) - d/ds  [px] = " << dpx_ds << " [py] = "
  // << dpy_ds << " [pz] = " << dpz_ds << "\n";
}

// ------------------------------------------------------------------------------------

/***********
template <typename MagField_t>
template <typename Real_t, typename Int_t>
inline
__host__ __device__ void
MagneticFieldEquation<MagField_t>::
   Force(Real_t const & momx, Real_t const & momy, Real_t const & momz,
         Real_t const & Bx,   Real_t const & By,   Real_t const & Bz,
         Int_t  const & charge,
         // Real_t const & momentumMag, //  Magnitude of initial momentum
         Real_t &  dx_ds, Real_t &  dy_ds, Real_t &  dz_ds,
         Real_t & dpx_ds, Real_t & dpy_ds, Real_t & dpz_ds
      )
{
   // using vecgeom::Vector3D<Real_t>;
   vecgeom::Vector3D<Real_t> Momentum = { momx, momy, momz } ;
   vecgeom::Vector3D<Real_t> Bfield = { Bx, By, Bz };
   // Real_t dydsArr[Nvar];
   Force( Momentum, charge, Bfield, dx_ds,  dy_ds, dz_ds, dpx_ds, dpy_ds, dpz_ds );

   // dx_ds=  dydsArr[0];    dy_ds= dydsArr[1];    dz_ds= dydsArr[2];
   // dpx_ds= dydsArr[3];   dpy_ds= dydsArr[4];   dpz_ds= dydsArr[5];
}
*************/

/**************************************************************
{
  Real_t inv_momentum_mag = Real_t(1.) /
     // vecCore::math::Sqrt
     std::sqrt
        (momx * momx + momy * momy + momz * momz);
  //    = vdt::fast_isqrt_general( momentum_sqr, 2); // Alternative

  dpx_ds = (momy * Bz - momz * By); //    Ax = a*(Vy*Bz - Vz*By)
  dpy_ds = (momz * Bx - momx * Bz); //    Ay = a*(Vz*Bx - Vx*Bz)
  dpz_ds = (momx * By - momy * Bx); //    Az = a*(Vx*By - Vy*Bx)

  Real_t cof = charge * Real_t(gCof) * inv_momentum_mag;

  dx_ds = momx * inv_momentum_mag; //  (d/ds)x = Vx/V
  dy_ds = momy * inv_momentum_mag; //  (d/ds)y = Vy/V
  dz_ds = momz * inv_momentum_mag; //  (d/ds)z = Vz/V

  dpx_ds *= cof;
  dpy_ds *= cof;
  dpz_ds *= cof;
  std::cout << "Force - mom = " << momx << " " << momy << " " << momz << " B= " << Bx << " " << By << " " << Bz << "\n";
  std::cout << "Force - d/ds  [px] = " << dpx_ds << " [py] = " << dpy_ds << " [pz] = " << dpz_ds << "\n";
}
************************************************************************/

// ------------------------------------------------------------------------------------

// template <typename MagField_t>
// template <typename Real_t>
// inline __host__ __device__ void MagneticFieldEquation<MagField_t>::EvaluateDerivatives(MagField_t const &magField,
//                                                                                        const Real_t y[], int charge,
//                                                                                        Real_t dy_ds[])
// {
//   const vecgeom::Vector3D<Real_t> Bvec = magField.Evaluate(y[0], y[1], y[2]);
//   EvaluateRhsGivenB<Real_t, int>(y, charge, Bvec, dy_ds);
// }

// ------------------------------------------------------------------------------------
// ONLY FOR DEBUGGING
template <typename MagField_t>
template <typename Real_t>
inline __host__ __device__ void MagneticFieldEquation<MagField_t>::EvaluateDerivativesGivenB(const Real_t y[],
                                                                                             const Real_t Bfield[3],
                                                                                             int charge, Real_t dy_ds[])
{
  const vecgeom::Vector3D<Real_t> Bvec = {Bfield[0], Bfield[1], Bfield[2]};
  EvaluateRhsGivenB<Real_t, int>(y, charge, Bvec, dy_ds);
}

// ------------------------------------------------------------------------------------
// ONLY FOR DEBUGGING
template <typename MagField_t>
template <typename Real_t>
inline __host__ __device__ void MagneticFieldEquation<MagField_t>::EvaluateDerivativesReturnB(
    MagField_t const &magField, const Real_t y[], int charge, Real_t dy_ds[], vecgeom::Vector3D<float> &BfieldVec
    // float           BfieldValue[3]
)
{
  BfieldVec = magField.Evaluate(y[0], y[1], y[2]);
  // std::cout << "EvalDerivRetB:  Bx= " << BfieldVec[0] << " By= " << BfieldVec[1] << " Bz=" << BfieldVec[2] <<
  // std::endl;
  EvaluateRhsGivenB<Real_t, int>(y, charge, BfieldVec, dy_ds);
}

// ------------------------------------------------------------------------------------
// ONLY FOR DEBUGGING
template <typename MagField_t>
template <typename Real_t>
inline __host__ __device__ void MagneticFieldEquation<MagField_t>::EvaluateDerivativesReturnB(
    MagField_t const &magField, vecgeom::Vector3D<float> const &position, vecgeom::Vector3D<float> const &momentum,
    int charge, Real_t dy_ds[], vecgeom::Vector3D<float> &BfieldVec)
{
  const Real_t y[Nvar] = {position[0], position[1], position[2], momentum[0], momentum[1], momentum[2]};
  BfieldVec            = magField.Evaluate(y[0], y[1], y[2]);
  // std::cout << "EvalDerivRetB:  Bx= " << BfieldVec[0] << " By= " << BfieldVec[1] << " Bz=" << BfieldVec[2] <<
  // std::endl;
  EvaluateRhsGivenB<Real_t, int>(y, charge, BfieldVec, dy_ds);
}

#endif
