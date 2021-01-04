// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/*
 * ConstBzFieldStepper.h
 *
 *  Copied/adapted from GeantV ConstBzFieldStepper.h
 *     J. Apostolakis,  11 Nov 2020
 * 
 *  Original Author: swenzel   ( S. Wenzel ) Apr 23, 2014
 */

#ifndef CONSTBzFIELDSTEPPER_H_
#define CONSTBzFIELDSTEPPER_H_

#include "VecGeom/base/Global.h"

//namespace adept {
// inline namespace ADEPT_IMPL_NAMESPACE {

/**
 * A very simple stepper treating the propagation of particles in a constant Bz magnetic field
 * ( neglecting energy loss of particle )
 * This class is roughly equivalent to TGeoHelix in ROOT
 */
class ConstBzFieldStepper
{

private:
  float fBz;    //   Adequate accuracy for all expected applications

public:
  VECCORE_ATT_HOST_DEVICE
  ConstBzFieldStepper(float Bz = 0.) : fBz(Bz) {}

  void   SetBz(float Bz) { fBz = Bz; }
  double GetBz() const { return fBz; }

  static constexpr float kB2C =
     -0.299792458 * (copcore::units::GeV /
                     (copcore::units::tesla * copcore::units::meter) ) ;
   
  /*
  template<typename RT, typename Vector3D>
  RT GetCurvature( Vector3D const & dir,
          double const charge, double const momentum ) const
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
  template <typename BaseType, typename BaseIType>
  inline __attribute__((always_inline))
  VECCORE_ATT_HOST_DEVICE
  void DoStep(
      BaseType const & /*posx*/, BaseType const & /*posy*/, BaseType const & /*posz*/, BaseType const & /*dirx*/,
      BaseType const & /*diry*/, BaseType const & /*dirz*/, BaseIType const & /*charge*/, BaseType const & /*momentum*/,
      BaseType const & /*step*/, BaseType & /*newsposx*/, BaseType & /*newposy*/, BaseType & /*newposz*/,
      BaseType & /*newdirx*/, BaseType & /*newdiry*/, BaseType & /*newdirz*/
      ) const;

  /**
   * this function propagates the track along the helix solution by a step
   * input: current position, current direction, some particle properties
   * output: new position, new direction of particle
   */
  template <typename Vector3D, typename BaseType, typename BaseIType>
  VECCORE_ATT_HOST_DEVICE
  void DoStep(Vector3D const &pos, Vector3D const &dir, BaseIType const &charge, BaseType const &momentum,
              BaseType const &step, Vector3D &newpos, Vector3D &newdir) const
  {
    DoStep(pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], charge, momentum, step, newpos[0], newpos[1], newpos[2],
           newdir[0], newdir[1], newdir[2]);
  }

}; // end class declaration


/**
 * this function propagates the track along the "helix-solution" by a step step
 * input: current position (x0, y0, z0), current direction ( dx0, dy0, dz0 ), some particle properties
 * output: new position, new direction of particle
 */
template <typename BaseDType, typename BaseIType>
inline __attribute__((always_inline)) 
void ConstBzFieldStepper::DoStep(
    BaseDType const &   x0,   BaseDType const &  y0,     BaseDType const &  z0, 
    BaseDType const &  dx0,   BaseDType const & dy0,     BaseDType const & dz0, 
    BaseIType const & charge, BaseDType const &momentum, BaseDType const &step, 
    BaseDType &  x, BaseDType &  y, BaseDType &  z, 
    BaseDType & dx, BaseDType & dy, BaseDType & dz
    ) const
{
  const double kSmall     = 1.E-30;
  // could do a fast square root here
  BaseDType dt      = sqrt((dx0 * dx0) + (dy0 * dy0)) + kSmall;
  BaseDType invnorm = 1. / dt;
  // radius has sign and determines the sense of rotation
  BaseDType R = momentum * dt / ((BaseDType(kB2C) * BaseDType(charge)) * (fBz));

  BaseDType cosa = dx0 * invnorm;
  BaseDType sina = dy0 * invnorm;
  BaseDType phi  = step * BaseDType(charge) * fBz * BaseDType(kB2C) / momentum;

  BaseDType cosphi;
  BaseDType sinphi;
  sincos(phi, &sinphi, &cosphi);

  x = x0 + R * (-sina - (-cosphi * sina - sinphi * cosa));
  y = y0 + R * (cosa - (-sinphi * sina + cosphi * cosa));
  z = z0 + step * dz0;

  dx = dx0 * cosphi - sinphi * dy0;
  dy = dx0 * sinphi + cosphi * dy0;
  dz = dz0;
}


#endif /* CONSTBzFIELDSTEPPER_H_ */
