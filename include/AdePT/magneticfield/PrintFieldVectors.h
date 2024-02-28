// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0
//
// Author:  J. Apostolakis,  19 Nov 2021
//
#ifndef PRINT_FIELD_VECTORS_H__
#define PRINT_FIELD_VECTORS_H__

static constexpr int NvarPrint = 6;

#include <iostream>
#include <iomanip>

#include "VecGeom/base/Vector3D.h"

namespace PrintFieldVectors {

constexpr int prec = 6;
constexpr int nd   = prec + 4;
// constexpr char* numFormat= "10.6f";

template <typename Real_t>
static inline __host__ __device__ void Print3vec(const Real_t y[3], const char *name)
{
  char numFormat[12];
  char format[64];

  sprintf(numFormat, "%%%d.%df", nd, prec);
  sprintf(format, " %%-12s  = %s %s %s ", numFormat, numFormat, numFormat);
  // std::cout << " Position             = " << setw(nd) << y[0] << " " << setw(nd) << y[1] << " " << setw(nd) << y[2];
  printf(format, name, y[0], y[1], y[2]);
}

template <typename Real_t>
static inline __host__ __device__ void PrintScalar(const Real_t yVal, const char *name[10])
{
  char numFormat[12];
  // char  format[64];

  sprintf(numFormat, "%%%d.%df", nd, prec);
  // sprintf( format, " %%s            = %s ", numFormat );
  // std::cout << " Position             = " << setw(nd) << y[0] << " " << setw(nd) << y[1] << " " << setw(nd) << y[2];
  // printf( format, name, yVal );
  printf("%12s", name);
  printf(numFormat, yVal);
}

template <typename Real_t>
static inline __host__ __device__ void PrintSixvec(const Real_t y[NvarPrint], Real_t originalMomentum = -1.0,
                                                   bool newline = true)
{
  char numFormat[12];
  sprintf(numFormat, "%%%d.%df", nd, prec);

  // printf( "\n# Six-Vector & B field \n");
  Print3vec(y, "Position");
  if (newline) {
    printf("\n");
  }
  Print3vec(&y[3], "Momentum");
  if (newline) {
    printf("\n");
  }

  if ((originalMomentum != -1.0) && (originalMomentum != 0.0)) {
    Real_t mag = sqrt(y[3] * y[3] + y[4] * y[4] + y[5] * y[5]);
    //  Vector3D<Real_t>( y[3], y[4], y[5]).Mag();
    // std::cout << " (|p|-o)/o = " << setw(nd) << (mag - originalMomentum) / originalMomentum;
    printf(" (|p|-o)/o = ");
    printf(numFormat, (mag - originalMomentum) / originalMomentum);
  }
  if (newline) printf("\n");
}

template <typename Real_t>
static inline void PrintSixvec(vecgeom::Vector3D<Real_t> const &position, vecgeom::Vector3D<Real_t> const &momentum,
                               Real_t origMomentumMag = -1.0)
{
  Real_t y[NvarPrint] = {position[0], position[1], position[2], momentum[0], momentum[1], momentum[2]};
  PrintSixVec(y, origMomentumMag);
}

template <typename Real_t>
static inline __host__ __device__ void PrintLineSixvec(const Real_t y[NvarPrint])
{
  Print3vec(y, "x:"); // std::cout << " x: " << setw(nd) << y[0] << " " << setw(nd) << y[1] << " " << setw(nd) << y[2];
  Print3vec(&y[3],
            "p:"); // std::cout << " p: " << setw(nd) << y[3] << " " << setw(nd) << y[4] << " " << setw(nd) << y[5];
  printf("\n");
}

template <typename Real_t>
static inline __host__ __device__ void PrintSixvecAndDyDx(const Real_t y[NvarPrint], int charge, const Real_t Bfield[3],
                                                          Real_t const dydx[NvarPrint])
{
  // RightHandSide<Real_t>(y, charge, dydx);

  // Obtain the field value
  // Vector3D<Real_t> Bfield;
  // FieldFromY(y, Bfield);
  // EvaluateRhsGivenB(y, charge, Bfield, dydx);

  PrintSixvec(y);
  // char bFieldName[12] = " B field [0-2]";
  Print3vec<float>(Bfield, /* bFieldName ); */ " B field [0-2]");
  printf("\n");

  // std::cout << "# 'Force' from B field \n";
  Print3vec(dydx, " dy/dx [0-2] (=dX/ds) = ");
  printf("\n");
  Print3vec(dydx + 3, " dy/dx [3-5] (=dP/ds) = ");
  printf("\n");
}

//  std::cout.unsetf(std::ios_base::scientific);

// template <typename Real_t> static inline // __host__ __device__
// void PrintSixvecAndDyDx(const Real_t y[NvarPrint], int charge, const Real_t Bfield[3], const Real_t dydx[NvarPrint])
// { }

template <typename Real_t>
static inline // __host__ __device__
    void
    PrintLineSixvecDyDx(const Real_t y[NvarPrint], int charge, const Real_t Bfield[3], Real_t const dydx[NvarPrint])
{
  using std::cout;
  using std::endl;
  using std::setw;
  constexpr int linPrec = 3; // Output precision - number of significant digits

  int oldprec = std::cout.precision(linPrec);

  PrintLineSixvec(y); // Was PrintSixvec( y, false );

  cout.setf(std::ios_base::fixed);
  // cout << " B field [0-2]        = ";
  // cout << " B: ";
  // cout << setw(nd) << Bfield[0] << " " << setw(nd) << Bfield[1] << " " << setw(nd) << Bfield[2];

  // cout << "# 'Force' from B field \n";
  // cout << " dy/dx [0-2] (=dX/ds) = ";
  cout << " dy_ds ";
  cout << setw(nd) << dydx[0] << " " << setw(nd) << dydx[1] << " " << setw(nd) << dydx[2] << " | ";
  // cout << " dy/dx [3-5] (=dP/ds) = ";
  cout << setw(nd + 2) << dydx[3] << " " << setw(nd + 2) << dydx[4] << " " << setw(nd + 2) << dydx[5];
  cout.unsetf(std::ios_base::fixed);
  cout << std::endl;
  cout.precision(oldprec);
}

//  cout.unsetf(std::ios_base::scientific);

// template <typename Real_t> static inline // __host__ __device__
// void PrintSixvecAndDyDx(const Real_t y[NvarPrint], int charge, const Real_t Bfield[3], const Real_t dydx[NvarPrint])
// { }

}; // namespace PrintFieldVectors

#endif
