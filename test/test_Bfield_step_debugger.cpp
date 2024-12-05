

#define __device__
#define __host__

#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/UniformMagneticField.h>
#include <AdePT/magneticfield/PrintFieldVectors.h>
#include <AdePT/magneticfield/fieldPropagatorRungeKutta.h>
#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/DormandPrinceRK45.h>

#include <AdePT/navigation/AdePTNavigator.h>

#include <VecGeom/base/Vector3D.h>

using Real_t = double;
using Field_t            = UniformMagneticField;
using MagFieldEquation_t = MagneticFieldEquation<Field_t>;

template <typename T>
using Vector3D = vecgeom::Vector3D<T>;

using Equation_t     = MagneticFieldEquation<Field_t>;
using Stepper_t      = DormandPrinceRK45<Equation_t, Field_t, 6, vecgeom::Precision>;
using RkDriver_t = RkIntegrationDriver<Stepper_t, vecgeom::Precision, int, Equation_t, Field_t>;

#include <iostream>


class TestPropagator : public fieldPropagatorRungeKutta<Field_t, RkDriver_t, Real_t, AdePTNavigator> {
public:
    using fieldPropagatorRungeKutta::IntegrateTrackToEnd;
};

int main(int argc, char **argv)
{
  using copcore::units::tesla;

  UniformMagneticField Bz(Vector3D<double>(0.0, 0.0, 3.8 * tesla));


  vecgeom::Vector3D<double> position =   {14.015712295645901, 121.660653344735479, -14505.751188880145492};
  vecgeom::Vector3D<double> momentumVec = {-0.000335427190908, -0.000002304326922, -0.000116722850709};


  double kinE = sqrt(momentumVec.Mag2() + copcore::units::kElectronMassC2* copcore::units::kElectronMassC2) - copcore::units::kElectronMassC2;
  printf("Ekin [MeV] = %.15f\n", kinE/copcore::units::MeV);
  double stepLength = 0.004715360602658;
  int charge = 1;
  int indx = 0;

  TestPropagator::IntegrateTrackToEnd(Bz, position, momentumVec, charge, stepLength, indx);

  return true;
}
