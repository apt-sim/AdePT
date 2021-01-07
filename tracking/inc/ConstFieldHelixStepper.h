// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/*
 * Created on: January 4, 2021
 * 
 * Adapted from GeantV/ConstFieldHelixStepper.h
 *
 *      Author: J. Apostolakis
 */

#ifndef CONSTFIELDHELIXSTEPPER_H_
#define CONSTFIELDHELIXSTEPPER_H_

#include "VecGeom/base/Global.h"
// Needed for Vector3D

// namespace adept {
// inline namespace ADEPT_IMPL_NAMESPACE {
/**
 * A simple stepper treating the propagation of particles in a constant magnetic field
 *   ( not just along the z-axis-- for that there is ConstBzFieldHelixStepper ) 
 */
class ConstFieldHelixStepper {
  template <typename T>
  using Vector3D = vecgeom::Vector3D<T>;

public:
  // VECCORE_ATT_HOST_DEVICE
  // ConstFieldHelixStepper(); // For default initialisation only
   
  VECCORE_ATT_HOST_DEVICE
  ConstFieldHelixStepper(float Bx, float By, float Bz);

  VECCORE_ATT_HOST_DEVICE
  ConstFieldHelixStepper(float Bfield[3]);

  VECCORE_ATT_HOST_DEVICE
  ConstFieldHelixStepper(Vector3D<float> const &Bfield);

  void
  VECCORE_ATT_HOST_DEVICE  
  SetB(float Bx, float By, float Bz)
  {
    // fB.Set(Bx, By, Bz);
    Vector3D<float> Bfield(Bx, By, Bz);
    CalculateDerived( Bfield );
  }

  VECCORE_ATT_HOST_DEVICE   
  Vector3D<float> const GetFieldVec() const { return fBmag * fUnit; }

  static constexpr float kB2C =
     -0.299792458 * (copcore::units::GeV /
                     (copcore::units::tesla * copcore::units::meter) ) ;
   
  /*
  template<typename RT, typename Vector3D>
  RT GetCurvature(Vector3D const & dir,
                  float const charge, float const momentum) const
  {
    if (charge == 0) return RT(0.);
    return abs( kB2C * fBz * dir.FastInverseScaledXYLength( momentum ) );
  }
  */

  /**
   * this function propagates the track along the helix solution by a step
   * input: current position, current direction, some particle properties
   * output: new position, new direction of particle
   */
  template <typename Real_t>
  VECCORE_ATT_HOST_DEVICE
  void DoStep(Real_t const &posx, Real_t const &posy, Real_t const &posz,
              Real_t const &dirx, Real_t const &diry, Real_t const &dirz,
              Real_t const &charge,
              Real_t const &momentum,
              Real_t const &step,
              Real_t &newposx, Real_t &newposy, Real_t &newposz,
              Real_t &newdirx, Real_t &newdiry, Real_t &newdirz
     ) const;
   
  /**
   * this function propagates the track along the helix solution by a step
   * input: current position, current direction, some particle properties
   * output: new position, new direction of particle
   */
  template <typename Real_t>
  inline   
  VECCORE_ATT_HOST_DEVICE
  void DoStep(Vector3D<Real_t> const & position,
              Vector3D<Real_t> const & direction,
              Real_t           const & charge,
              Real_t           const & momentum,
              Real_t           const & step,
              Vector3D<Real_t>       & endPosition,
              Vector3D<Real_t>       & endDirection) const;
   
  // Auxiliary methods
  template <typename Real_t>
   void PrintStep(vecgeom::Vector3D<Real_t> const &startPosition,
                                    vecgeom::Vector3D<Real_t> const &startDirection, Real_t const &charge,
                                    Real_t const &momentum, Real_t const &step, vecgeom::Vector3D<Real_t> &endPosition,
                                    vecgeom::Vector3D<Real_t> &endDirection) const;

protected:
  inline VECCORE_ATT_HOST_DEVICE
  void CalculateDerived( Vector3D<float> Bvec );

  template <typename Real_t>
  inline  VECCORE_ATT_HOST_DEVICE
  bool CheckModulus(Real_t &newdirX_v, Real_t &newdirY_v, Real_t &newdirZ_v) const;

private:
  // Values below used for speed, code simplicity
  Vector3D<float> fUnit = Vector3D<float>(1.0, 0.0, 0.0);   // direction
  float           fBmag = 0.0;                              // magnitude
   
}; // end class declaration

inline // __host__ __device__
void ConstFieldHelixStepper::CalculateDerived(  Vector3D<float> Bvec )
{
  fBmag = Bvec.Mag();
  float  bMagInv = (1.0/fBmag);
  fUnit = Vector3D<float>(1.0, 0.0, 0.0);
  if( fBmag > 0.0 )
     fUnit = bMagInv * Bvec;
}

inline
ConstFieldHelixStepper::ConstFieldHelixStepper(float Bx, float By, float Bz) // : fB(Bx, By, gBz)
{
  CalculateDerived( Vector3D<float>(Bx, By, Bz) );
}

inline
ConstFieldHelixStepper::ConstFieldHelixStepper(float B[3]) // : fB(B[0], B[1], B[2])
{
  CalculateDerived( Vector3D<float>(B[0], B[1], B[2]) );  
}

inline
VECCORE_ATT_HOST_DEVICE
ConstFieldHelixStepper::ConstFieldHelixStepper(vecgeom::Vector3D<float> const &Bfield)  // : fB(Bfield)
{
  CalculateDerived( Bfield );
}

/**
 * this function propagates the track along the "helix-solution" by a step step
 * input: current position (x0, y0, z0), current direction ( dirX0, dirY0, dirZ0 ), some particle properties
 * output: new position, new direction of particle
 */
template <typename Real_t>
inline void ConstFieldHelixStepper::DoStep(Real_t const &x0, Real_t const &y0, Real_t const &z0,
                                           Real_t const &dirX0, Real_t const &dirY0, Real_t const &dirZ0,
                                           Real_t const &charge, Real_t const &momentum, Real_t const &step,
                                           Real_t &x,  Real_t &y,  Real_t &z,
                                           Real_t &dx, Real_t &dy, Real_t &dz) const
                                           
{
  vecgeom::Vector3D<Real_t> startPosition(x0, y0, z0);
  vecgeom::Vector3D<Real_t> startDirection(dirX0, dirY0, dirZ0);
  vecgeom::Vector3D<Real_t> endPosition, endDirection;

  // startPosition.Set( x0, y0, z0);
  // startDirection.Set( dirX0, dirY0, dirZ0);

  DoStep(startPosition, startDirection, charge, momentum, step, endPosition, endDirection);
  x  = endPosition.x();
  y  = endPosition.y();
  z  = endPosition.z();
  dx = endDirection.x();
  dy = endDirection.y();
  dz = endDirection.z();

  // PrintStep(startPosition, startDirection, charge, momentum, step, endPosition, endDirection);
}

template <typename Real_t>
inline 
VECCORE_ATT_HOST_DEVICE
void ConstFieldHelixStepper::DoStep(vecgeom::Vector3D<Real_t> const &startPosition,
                                    vecgeom::Vector3D<Real_t> const &startDirection,
                                    Real_t const &charge,
                                    Real_t const &momentum,
                                    Real_t const &step,
                                    vecgeom::Vector3D<Real_t> &endPosition,
                                    vecgeom::Vector3D<Real_t> &endDirection) const
{
  // const Real_t kB2C_local(-0.299792458e-3);
  const Real_t kSmall(1.E-30);
  using vecgeom::Vector3D;

  // std::cout << " ConstFieldHelixStepper::DoStep called.  fBmag= " << fBmag
  //          << " unit dir= " << fUnit << std::endl;

  // assert( std::abs( startDirection.Mag2() - 1.0 ) < 1.0e-6 );

  Vector3D<Real_t> dir1Field(fUnit);
  Real_t UVdotUB = startDirection.Dot(dir1Field); //  Limit cases 0.0 and 1.0
  Real_t dt2     = max(startDirection.Mag2() - UVdotUB * UVdotUB, Real_t(0.0));
  Real_t sinVB   = sqrt(dt2) + kSmall;

  // radius has sign and determines the sense of rotation
  Real_t R = momentum * sinVB / (kB2C * charge * fBmag);

  Vector3D<Real_t> restVelX = startDirection - UVdotUB * dir1Field;

  Vector3D<Real_t> dirVelX    = restVelX.Unit();          // Unit must cope with 0 length !!
  Vector3D<Real_t> dirCrossVB = dirVelX.Cross(dir1Field); // OK if it is zero

  /***
  printf("\n");
  printf("CVFHS> dir-1  B-fld  = %f %f %f   mag-1= %g \n", dir1Field.x(), dir1Field.y(), dir1Field.z(),
         dir1Field.Mag()-1.0 );
  printf("CVFHS> dir-2  VelX   = %f %f %f   mag-1= %g \n", dirVelX.x(), dirVelX.y(), dirVelX.z(),
         dirVelX.Mag()-1.0 );
  printf("CVFHS> dir-3: CrossVB= %f %f %f   mag-1= %g \n", dirCrossVB.x(), dirCrossVB.y(), dirCrossVB.z(),
         dirCrossVB.Mag()-1.0 );
  // dirCrossVB = dirCrossVB.Unit();
  printf("CVFHS> Dot products   d1.d2= %g   d2.d3= %g  d3.d1= %g \n",
         dir1Field.Dot(dirVelX), dirVelX.Dot( dirCrossVB), dirCrossVB.Dot(dir1Field) );
   ***/
  assert(fabs(dir1Field.Dot(dirVelX)) < 1.e-6);
  assert(fabs(dirVelX.Dot(dirCrossVB)) < 1.e-6);
  assert(fabs(dirCrossVB.Dot(dir1Field)) < 1.e-6);

  Real_t phi = -step * charge * fBmag * kB2C / momentum;

  // printf("CVFHS> phi= %g \n", vecCore::Get(phi,0) );  // phi (scalar)  or phi[0] (vector)

  Real_t cosphi;              //  = cos(phi);
  Real_t sinphi;              //  = sin(phi);
  sincos(phi, &sinphi, &cosphi);

  endPosition = startPosition + R * (cosphi - 1) * dirCrossVB - R * sinphi * dirVelX +
                step * UVdotUB * dir1Field; //   'Drift' along field direction

  endDirection = UVdotUB * dir1Field + cosphi * sinVB * dirVelX + sinphi * sinVB * dirCrossVB;
}

//________________________________________________________________________________
template <typename Real_t>
bool
ConstFieldHelixStepper::CheckModulus(Real_t &newdirX_v,
                                     Real_t &newdirY_v,
                                     Real_t &newdirZ_v) const
{
  constexpr float perMillion = 1.0e-6;

  Real_t modulusDir = newdirX_v * newdirX_v + newdirY_v * newdirY_v + newdirZ_v * newdirZ_v;
  typename vecCore::Mask<Real_t> goodDir;
  goodDir = vecCore::math::Abs(modulusDir - Real_t(1.0)) < perMillion;

  bool allGood = vecCore::MaskFull(goodDir);
  assert(allGood && "Not all Directions are nearly 1");

  return allGood;
}


template <typename Real_t>
inline
void ConstFieldHelixStepper::PrintStep(
   vecgeom::Vector3D<Real_t> const &startPosition,
   vecgeom::Vector3D<Real_t> const &startDirection,
   Real_t const &charge,
   Real_t const &momentum,
   Real_t const &step,
   vecgeom::Vector3D<Real_t> &endPosition,
   vecgeom::Vector3D<Real_t> &endDirection) const
{
  // Debug printing of input & output
  printf(" HelixSteper::PrintStep \n");
  const int vectorSize = vecCore::VectorSize<Real_t>();
  Real_t x0, y0, z0, dirX0, dirY0, dirZ0;
  Real_t x, y, z, dx, dy, dz;
  x0    = startPosition.x();
  y0    = startPosition.y();
  z0    = startPosition.z();
  dirX0 = startDirection.x();
  dirY0 = startDirection.y();
  dirZ0 = startDirection.z();
  x     = endPosition.x();
  y     = endPosition.y();
  z     = endPosition.z();
  dx    = endDirection.x();
  dy    = endDirection.y();
  dz    = endDirection.z();
  for (int i = 0; i < vectorSize; i++) {
    printf("Start> Lane= %1d Pos= %8.5f %8.5f %8.5f  Dir= %8.5f %8.5f %8.5f ",
           i, x0, y0, z0,
           dirX0, dirY0, dirZ0);
    printf(" s= %10.6f ", step );     // / units::mm );
    printf(" q= %3.1f ", charge );    // in e+ units ?
    printf(" p= %10.6f ", momentum ); // / units::GeV );
    // printf(" ang= %7.5f ", angle );
    printf(" End> Pos= %9.6f %9.6f %9.6f  Mom= %9.6f %9.6f %9.6f\n", x, y, z, dx,
           dy, dz);
  }
}

// } // namespace ADEPT_IMPL_NAMESPACE
// } // namespace adept

#endif /* CONSTFIELDHELIXSTEPPER_H_ */
