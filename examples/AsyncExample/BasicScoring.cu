// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "BasicScoring.h"
#include "AdeptIntegration.h"

#include <CopCore/Global.h>
#include <CopCore/PhysicalConstants.h>

#include "Track.cuh" // not nice - we expose the track model here, interface of DepositEnergy to be changed

#include <iostream>
#include <iomanip>
#include <stdio.h>

void BasicScoring::InitializeOnGPU(BasicScoring *const BasicScoring_dev)
{
  fAuxData_dev = AdeptIntegration::VolAuxArray::GetInstance().fAuxData_dev;
  // Allocate memory to score charged track length and energy deposit per volume.
  COPCORE_CUDA_CHECK(cudaMalloc(&fChargedTrackLength_dev, sizeof(double) * fNumSensitive));
  COPCORE_CUDA_CHECK(cudaMalloc(&fEnergyDeposit_dev, sizeof(double) * fNumSensitive));

  // Allocate and initialize scoring and statistics.
  COPCORE_CUDA_CHECK(cudaMalloc(&fGlobalScoring_dev, sizeof(GlobalScoring)));

  ScoringPerVolume scoringPerVolume_devPtrs;
  scoringPerVolume_devPtrs.chargedTrackLength = fChargedTrackLength_dev;
  scoringPerVolume_devPtrs.energyDeposit      = fEnergyDeposit_dev;
  COPCORE_CUDA_CHECK(cudaMalloc(&fScoringPerVolume_dev, sizeof(ScoringPerVolume)));
  COPCORE_CUDA_CHECK(
      cudaMemcpy(fScoringPerVolume_dev, &scoringPerVolume_devPtrs, sizeof(ScoringPerVolume), cudaMemcpyHostToDevice));

  // Now copy host instance to device
  COPCORE_CUDA_CHECK(cudaMemcpy(BasicScoring_dev, this, sizeof(BasicScoring), cudaMemcpyHostToDevice));

  ClearGPU();
}

BasicScoring::BasicScoring(int numSensitive)
    : fNumSensitive{numSensitive}, fChargedTrackLength{new double[numSensitive]},
      fEnergyDeposit{new double[numSensitive]}, fScoringPerVolume{fEnergyDeposit, fChargedTrackLength}
{
}

/// @brief Copy the host side of the scoring. The device side will remain uninitialised.
BasicScoring::BasicScoring(const BasicScoring &other)
    : fNumSensitive{other.fNumSensitive}, fChargedTrackLength{new double[fNumSensitive]},
      fEnergyDeposit{new double[fNumSensitive]}, fScoringPerVolume{fEnergyDeposit, fChargedTrackLength},
      fGlobalScoring{other.fGlobalScoring}
{
  std::copy(other.fChargedTrackLength, other.fChargedTrackLength + fNumSensitive, fChargedTrackLength);
  std::copy(other.fEnergyDeposit, other.fEnergyDeposit + fNumSensitive, fEnergyDeposit);
  // Only one instance can own the device pointers:
  fEnergyDeposit_dev      = nullptr;
  fChargedTrackLength_dev = nullptr;
  fScoringPerVolume_dev   = nullptr;
  fGlobalScoring_dev      = nullptr;
}

/// @brief Move the scoring. The moved-to instance will own the device pointers.
BasicScoring::BasicScoring(BasicScoring &&other)
    : fNumSensitive{other.fNumSensitive}, fAuxData_dev{other.fAuxData_dev},
      fEnergyDeposit_dev{other.fEnergyDeposit_dev}, fChargedTrackLength_dev{other.fChargedTrackLength_dev},
      fScoringPerVolume_dev{other.fScoringPerVolume_dev}, fGlobalScoring_dev{other.fGlobalScoring_dev},
      fChargedTrackLength{std::move(other.fChargedTrackLength)}, fEnergyDeposit{std::move(other.fEnergyDeposit)},
      fScoringPerVolume{std::move(other.fScoringPerVolume)}, fGlobalScoring{std::move(other.fGlobalScoring)}
{
  // Only one instance can own the device pointers:
  other.fEnergyDeposit_dev      = nullptr;
  other.fChargedTrackLength_dev = nullptr;
  other.fScoringPerVolume_dev   = nullptr;
  other.fGlobalScoring_dev      = nullptr;
  other.fChargedTrackLength     = nullptr;
  other.fEnergyDeposit          = nullptr;
}

BasicScoring::~BasicScoring()
{
  FreeGPU();
  delete[] fChargedTrackLength;
  delete[] fEnergyDeposit;
}

void BasicScoring::FreeGPU()
{
  // Free resources.
  COPCORE_CUDA_CHECK(cudaFree(fChargedTrackLength_dev));
  COPCORE_CUDA_CHECK(cudaFree(fEnergyDeposit_dev));

  COPCORE_CUDA_CHECK(cudaFree(fGlobalScoring_dev));
  COPCORE_CUDA_CHECK(cudaFree(fScoringPerVolume_dev));
}

void BasicScoring::ClearGPU()
{
  // Clear the device hits content
  COPCORE_CUDA_CHECK(cudaMemset(fGlobalScoring_dev, 0, sizeof(GlobalScoring)));
  COPCORE_CUDA_CHECK(cudaMemset(fChargedTrackLength_dev, 0, sizeof(double) * fNumSensitive));
  COPCORE_CUDA_CHECK(cudaMemset(fEnergyDeposit_dev, 0, sizeof(double) * fNumSensitive));
}

void BasicScoring::CopyHitsToHost()
{
  // Transfer back scoring.
  COPCORE_CUDA_CHECK(cudaMemcpy(&fGlobalScoring, fGlobalScoring_dev, sizeof(GlobalScoring), cudaMemcpyDeviceToHost));

  // Transfer back the scoring per volume (charged track length and energy deposit).
  COPCORE_CUDA_CHECK(cudaMemcpy(fScoringPerVolume.chargedTrackLength, fChargedTrackLength_dev,
                                sizeof(double) * fNumSensitive, cudaMemcpyDeviceToHost));
  COPCORE_CUDA_CHECK(cudaMemcpy(fScoringPerVolume.energyDeposit, fEnergyDeposit_dev, sizeof(double) * fNumSensitive,
                                cudaMemcpyDeviceToHost));
}

__device__ void BasicScoring::Score(vecgeom::NavStateIndex const &crt_state, int charge, double geomStep, double edep)
{
  assert(fGlobalScoring_dev && "Scoring not initialized on device");
  auto volume  = crt_state.Top();
  int volumeID = volume->id();
  int charged  = abs(charge);

  int lvolID = volume->GetLogicalVolume()->id();

  // Add to charged track length, global energy deposit and deposit per volume
  atomicAdd(&fScoringPerVolume_dev->chargedTrackLength[volumeID], charged * geomStep);
  atomicAdd(&fGlobalScoring_dev->energyDeposit, edep);
  atomicAdd(&fScoringPerVolume_dev->energyDeposit[volumeID], edep);
}

__device__ void BasicScoring::AccountHit()
{
  // Increment hit counter
  atomicAdd(&fGlobalScoring_dev->hits, 1);
}

__device__ void BasicScoring::AccountChargedStep(int charge)
{
  // Increase counters for charged/neutral steps
  int charged = abs(charge);
  // Increment global number of steps
  atomicAdd(&fGlobalScoring_dev->chargedSteps, charged);
  atomicAdd(&fGlobalScoring_dev->neutralSteps, 1 - charged);
}

__device__ void BasicScoring::AccountProduced(int num_ele, int num_pos, int num_gam)
{
  // Increment number of secondaries
  atomicAdd(&fGlobalScoring_dev->numElectrons, num_ele);
  atomicAdd(&fGlobalScoring_dev->numPositrons, num_pos);
  atomicAdd(&fGlobalScoring_dev->numGammas, num_gam);
}
