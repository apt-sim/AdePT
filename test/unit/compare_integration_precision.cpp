// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  N. Misyats, 05 Aug 2025
//

#define __device__
#define __host__

#include <VecGeom/base/Vector3D.h>
#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/UniformMagneticField.h>
#include <AdePT/magneticfield/PrintFieldVectors.h>
#include <AdePT/magneticfield/ConstFieldHelixStepper.h>
#include <AdePT/magneticfield/DormandPrinceRK45.h>
#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/magneticfield/RkIntegrationDriver.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>

constexpr unsigned int Nvar = 6; // Number of integration variables
constexpr int charge        = -1;

// using namespace vecCore::math;

template <typename T>
using vec3 = vecgeom::Vector3D<T>;

template <typename Real_t>
struct IntegrationPoint {
  vec3<Real_t> position;
  vec3<Real_t> momentum;
  Real_t distance;

  vec3<Real_t> GetDirection() const { return momentum / momentum.Length(); }

  Real_t GetMomentumMag() const { return momentum.Length(); }
};

template <typename Real_t>
void IntegrateForDistance(const UniformMagneticField &magField, const Real_t maxDistance,
                          const IntegrationPoint<Real_t> &initialState,
                          std::vector<IntegrationPoint<Real_t>> &pointsOut, Real_t minStepSize, int maxNumberOfSteps)
{

  using Equation_t = MagneticFieldEquation<UniformMagneticField>;
  using Stepper_t  = DormandPrinceRK45<Equation_t, UniformMagneticField, Nvar, Real_t>;
  using Driver_t   = RkIntegrationDriver<Stepper_t, Real_t, int, Equation_t, UniformMagneticField>;

  Real_t totalDistanceDone = initialState.distance;
  Real_t yCurrentIn[Nvar];
  Real_t dydxCurrentIn[Nvar];
  Real_t yNextOut[Nvar];
  Real_t dydxNextOut[Nvar];

  Real_t stepSizeTryIn = minStepSize;
  Real_t stepSizeNextOut;

  yCurrentIn[0] = initialState.position[0];
  yCurrentIn[1] = initialState.position[1];
  yCurrentIn[2] = initialState.position[2];
  yCurrentIn[3] = initialState.momentum[0];
  yCurrentIn[4] = initialState.momentum[1];
  yCurrentIn[5] = initialState.momentum[2];

  Equation_t::EvaluateDerivatives(magField, yCurrentIn, charge, dydxCurrentIn);

  int numSteps = 0;
  bool done    = false;
  pointsOut.push_back(initialState);
  do {
    bool goodStep = Driver_t::IntegrateStep(yCurrentIn, dydxCurrentIn, charge, totalDistanceDone, stepSizeTryIn,
                                            magField, yNextOut, dydxNextOut, stepSizeNextOut);

    done          = (totalDistanceDone >= maxDistance);
    stepSizeTryIn = vecCore::Max(minStepSize, stepSizeNextOut);
    if (goodStep && !done) {
      for (unsigned int i = 0; i < Nvar; i++) {
        yCurrentIn[i]    = yNextOut[i];
        dydxCurrentIn[i] = dydxNextOut[i]; // Using FSAL property !
      }
      stepSizeTryIn = vecCore::Min(stepSizeTryIn, maxDistance - totalDistanceDone);
    }

    IntegrationPoint<Real_t> currentPoint;
    currentPoint.position[0] = yCurrentIn[0];
    currentPoint.position[1] = yCurrentIn[1];
    currentPoint.position[2] = yCurrentIn[2];
    currentPoint.momentum[0] = yCurrentIn[3];
    currentPoint.momentum[1] = yCurrentIn[4];
    currentPoint.momentum[2] = yCurrentIn[5];
    currentPoint.distance    = totalDistanceDone;
    pointsOut.push_back(currentPoint);

    ++numSteps;

  } while (!done && numSteps < maxNumberOfSteps);
}

template <typename Real_t>
Real_t Random01()
{
  return ((Real_t)rand()) / ((Real_t)RAND_MAX);
}

template <typename Real_t>
vec3<Real_t> RandomUnitVec3()
{
  Real_t x = Random01<Real_t>() * 2.0 - 1.0;
  Real_t y = Random01<Real_t>() * 2.0 - 1.0;
  Real_t z = Random01<Real_t>() * 2.0 - 1.0;
  vec3<Real_t> vec(x, y, z);
  vec = vec / vec.Length();
  return vec;
}

template <typename Real_t>
void WritePointsToFile(std::ofstream &f, const std::vector<IntegrationPoint<Real_t>> &points, const std::string &label)
{
  f << label << std::endl << points.size() << std::endl;
  for (auto &point : points) {
    f << point.position[0] << " " << point.position[1] << " " << point.position[2] << " " << point.momentum[0] << " "
      << point.momentum[1] << " " << point.momentum[2] << " " << point.distance << std::endl;
  }
}

int main(int argc, char **argv)
{
  // srand(0);
  srand(time({}));

  constexpr int numRepeats = 10000;

  double maxDistance   = 10000.0;
  double minStepSize   = 1e-4;
  int maxNumberOfSteps = 1000000;

  std::ofstream fileDouble("./points_double.txt");
  std::ofstream fileFloat("./points_float.txt");

  for (int n = 0; n < numRepeats; ++n) {
    double magFieldMag                   = 0.1 + (4.0 - 0.1) * Random01<double>();
    const vec3<double> magFieldVecDouble = RandomUnitVec3<double>() * magFieldMag * (double)copcore::units::tesla;
    UniformMagneticField magFieldDouble(magFieldVecDouble);
    IntegrationPoint<double> initialStateDouble;
    double startMomentumMag     = 1.0 + (10000.0 - 1.0) * Random01<double>();
    initialStateDouble.position = RandomUnitVec3<double>() * 1000.0;
    initialStateDouble.momentum = RandomUnitVec3<double>() * startMomentumMag * (double)copcore::units::MeV;
    initialStateDouble.distance = 0.0f;

    std::cout << "Magnetic field: " << magFieldVecDouble.Length() << " T" << std::endl
              << "  x = " << magFieldVecDouble[0] << std::endl
              << "  y = " << magFieldVecDouble[1] << std::endl
              << "  z = " << magFieldVecDouble[2] << std::endl;
    std::cout << "Initial position:" << std::endl
              << "  x = " << initialStateDouble.position[0] << std::endl
              << "  y = " << initialStateDouble.position[1] << std::endl
              << "  z = " << initialStateDouble.position[2] << std::endl;
    std::cout << "Initial momentum: " << initialStateDouble.momentum.Length() << " MeV" << std::endl
              << "  x = " << initialStateDouble.momentum[0] << std::endl
              << "  y = " << initialStateDouble.momentum[1] << std::endl
              << "  z = " << initialStateDouble.momentum[2] << std::endl;

    // Integrate at double precision from the initial state
    std::vector<IntegrationPoint<double>> pointsDouble;
    IntegrateForDistance(magFieldDouble, (double)maxDistance, initialStateDouble, pointsDouble, (double)minStepSize,
                         maxNumberOfSteps);

    // Convert the same quantities to float
    const vec3<float> magFieldVecFloat((float)magFieldVecDouble[0], (float)magFieldVecDouble[1],
                                       (float)magFieldVecDouble[2]);
    UniformMagneticField magFieldFloat(magFieldVecFloat);
    IntegrationPoint<float> initialStateFloat;
    initialStateFloat.position =
        vec3<float>((float)initialStateDouble.position[0], (float)initialStateDouble.position[1],
                    (float)initialStateDouble.position[2]);
    initialStateFloat.momentum =
        vec3<float>((float)initialStateDouble.momentum[0], (float)initialStateDouble.momentum[1],
                    (float)initialStateDouble.momentum[2]);
    initialStateFloat.distance = (float)initialStateDouble.distance;

    // Integrate at float precision from the initial state
    std::vector<IntegrationPoint<float>> pointsFloat;
    IntegrateForDistance(magFieldFloat, (float)maxDistance, initialStateFloat, pointsFloat, (float)minStepSize,
                         maxNumberOfSteps);

    std::cout << "Integration steps double: " << pointsDouble.size() - 1 << std::endl;
    std::cout << "Integration steps float: " << pointsFloat.size() - 1 << std::endl;
    std::cout << "Final distance double: " << pointsDouble.back().distance << std::endl;
    std::cout << "Final distance float: " << pointsFloat.back().distance << std::endl;

    // Evaluate analytical helix at the positions of both double and float integration
    std::vector<IntegrationPoint<double>> pointsHxDouble;
    std::vector<IntegrationPoint<float>> pointsHxFloat;
    for (const IntegrationPoint<double> &pointDouble : pointsDouble) {
      IntegrationPoint<double> pointHxDouble;
      ConstFieldHelixStepper helixDouble(magFieldVecDouble);
      helixDouble.DoStep(initialStateDouble.position, initialStateDouble.GetDirection(), charge,
                         initialStateDouble.GetMomentumMag(), pointDouble.distance, pointHxDouble.position,
                         pointHxDouble.momentum);
      pointHxDouble.momentum *= initialStateDouble.GetMomentumMag();
      pointHxDouble.distance = pointDouble.distance;
      pointsHxDouble.push_back(pointHxDouble);
    }
    for (const IntegrationPoint<float> &pointFloat : pointsFloat) {
      IntegrationPoint<float> pointHxFloat;
      ConstFieldHelixStepper helixFloat(magFieldVecFloat);
      helixFloat.DoStep(initialStateFloat.position, initialStateFloat.GetDirection(), charge,
                        initialStateFloat.GetMomentumMag(), pointFloat.distance, pointHxFloat.position,
                        pointHxFloat.momentum);
      pointHxFloat.momentum *= initialStateFloat.GetMomentumMag();
      pointHxFloat.distance = pointFloat.distance;
      pointsHxFloat.push_back(pointHxFloat);
    }

    std::stringstream refLabel, tstLabel;
    refLabel << "points_" << n << "_ref";
    tstLabel << "points_" << n << "_test";
    WritePointsToFile<double>(fileDouble, pointsHxDouble, refLabel.str());
    WritePointsToFile<double>(fileDouble, pointsDouble, tstLabel.str());
    WritePointsToFile<float>(fileFloat, pointsHxFloat, refLabel.str());
    WritePointsToFile<float>(fileFloat, pointsFloat, tstLabel.str());
  }
}
