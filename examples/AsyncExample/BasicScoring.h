// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef SCORING_H
#define SCORING_H

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>
#include <VecGeom/navigation/NavStateIndex.h>
#include "CommonStruct.h"

struct BasicScoring;
using AdeptScoring = BasicScoring;

// Data structures for scoring. The accessors must make sure to use atomic operations if needed.
struct GlobalScoring {
  double energyDeposit;
  // Not int to avoid overflows for more than 100,000 events; unsigned long long
  // is the only other data type available for atomicAdd().
  unsigned long long chargedSteps;
  unsigned long long neutralSteps;
  unsigned long long hits;
  unsigned long long numGammas;
  unsigned long long numElectrons;
  unsigned long long numPositrons;
  // Not used on the device, filled in by the host.
  unsigned long long numKilled;

  void Print()
  {
    printf("Global scoring: stpChg=%llu stpNeu=%llu hits=%llu numGam=%llu numEle=%llu numPos=%llu numKilled=%llu\n",
           chargedSteps, neutralSteps, hits, numGammas, numElectrons, numPositrons, numKilled);
  }
};

struct ScoringPerVolume {
  double *energyDeposit;
  double *chargedTrackLength;
};

struct BasicScoring {
  using VolAuxData = adeptint::VolAuxData;
  int fNumSensitive{0};
  VolAuxData *fAuxData_dev{nullptr};
  double *fEnergyDeposit_dev{nullptr};
  double *fChargedTrackLength_dev{nullptr};
  ScoringPerVolume *fScoringPerVolume_dev{nullptr};
  GlobalScoring *fGlobalScoring_dev{nullptr};

  double *fChargedTrackLength{nullptr};
  double *fEnergyDeposit{nullptr};
  ScoringPerVolume fScoringPerVolume;
  GlobalScoring fGlobalScoring;

  BasicScoring(int numSensitive) : fNumSensitive(numSensitive)
  {
    fEnergyDeposit                       = new double[numSensitive];
    fChargedTrackLength                  = new double[numSensitive];
    fScoringPerVolume.energyDeposit      = fEnergyDeposit;
    fScoringPerVolume.chargedTrackLength = fChargedTrackLength;
  }

  ~BasicScoring()
  {
    delete[] fEnergyDeposit;
    delete[] fChargedTrackLength;
  }

  __device__ __forceinline__ VolAuxData const &GetAuxData_dev(int volId) const { return fAuxData_dev[volId]; }

  /// @brief Simple step+edep scoring interface.
  __device__ void Score(vecgeom::NavStateIndex const &crt_state, int charge, double geomStep, double edep);

  /// @brief Account for a single hit
  __device__ void AccountHit();

  /// @brief Account for a charged step
  __device__ void AccountChargedStep(int charge);

  /// @brief Account for the number of produced secondaries
  __device__ void AccountProduced(int num_ele, int num_pos, int num_gam);

  /// @brief Initialize hit data structures on device
  BasicScoring *InitializeOnGPU();

  /// @brief Copy hits to host for a single event
  void CopyHitsToHost();

  /// @brief Clear hits on device to reuse for next event
  void ClearGPU();

  /// @brief Free data structures allocated on GPU
  void FreeGPU();

  /// @brief Print scoring info
  void Print() { fGlobalScoring.Print(); };
};

#endif
