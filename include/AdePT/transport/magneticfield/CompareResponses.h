// SPDX-FileCopyrightText: 2020-3 CERN
// SPDX-License-Identifier: Apache-2.0

// Author:   J. Apostolakis,  22 June 2022

#ifndef CompareResponses_hh
#define CompareResponses_hh 1

#include <VecGeom/base/Vector3D.h>

static __device__ __host__ bool CompareResponseVector3D(
    int id, vecgeom::Vector3D<double> const &originalVec, vecgeom::Vector3D<double> const &baselineVec,
    vecgeom::Vector3D<double> const &resultVec, // Output of new method
    const char *vecName,
    double thresholdRel // fraction difference allowed
)
// Returns 'true' if values are 'bad'...
{
  bool bad                           = false; // Good ..
  double magOrig                     = originalVec.Mag();
  vecgeom::Vector3D<double> moveBase = baselineVec - originalVec;
  vecgeom::Vector3D<double> moveRes  = resultVec - originalVec;
  double magMoveBase                 = moveBase.Mag();
  double magDiffRes                  = moveRes.Mag();

  if (std::fabs(magDiffRes / magMoveBase) - 1.0 > thresholdRel ||
      (resultVec - baselineVec).Mag() > thresholdRel * magMoveBase) {
    // printf("Difference seen in vector %s : ", vecName );
    printf("\n id %3d - Diff in %s: "
           " new-base= %14.9g %14.9g %14.9g (mag= %14.9g) "
           " mv_Res/mv_Base-1 = %7.3g | mv/base: 3v= %14.9g %14.9g %14.9g (mag= %9.4g)"
           " | mv/new: 3v= %14.9g %14.9g %14.9g (mag = %14.9g)"
           " || origVec= %14.9f %14.9f %14.9f (mag=%14.9f) | base= %14.9f %14.9f %14.9f (mag=%9.4g) \n",
           id, vecName, resultVec[0] - baselineVec[0], resultVec[1] - baselineVec[1], resultVec[2] - baselineVec[2],
           (resultVec - baselineVec).Mag(), (moveRes.Mag() / moveBase.Mag() - 1.0), moveBase[0], moveBase[1],
           moveBase[2], moveBase.Mag(),
           //      printf("   new-original: mag= %20.16g ,  new_vec= %14.9f , %14.9f , %14.9f \n",
           moveRes[0], moveRes[1], moveRes[2], moveRes.Mag(), originalVec[0], originalVec[1], originalVec[2],
           originalVec.Mag(), baselineVec[0], baselineVec[1], baselineVec[2], baselineVec.Mag());
    bad = true;
  }
  return bad;
};

static __device__ __host__ void ReportSameMoveVector3D(int id, vecgeom::Vector3D<double> const &originalVec,
                                                       vecgeom::Vector3D<double> const &resultVec, const char *vecName)
{
  printf(" id %3d - Same %s: "
         " mv/base: 3v= %14.9f %14.9f %14.9f (mag= %9.4g)"
         " || origVec= %14.9f %14.9f %14.9f (mag=%14.9f) | result= %14.9f %14.9f %14.9f (mag=%9.4g) \n",
         id, vecName, resultVec[0] - originalVec[0], resultVec[1] - originalVec[1], resultVec[2] - originalVec[2],
         (resultVec - originalVec).Mag(), originalVec[0], originalVec[1], originalVec[2], originalVec.Mag(),
         resultVec[0], resultVec[1], resultVec[2], resultVec.Mag());
}

#endif /* CompareResponses_hh */
