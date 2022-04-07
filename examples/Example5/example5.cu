// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example5.h"

#include <CopCore/Global.h>
#include <CopCore/SystemOfUnits.h>
#include <CopCore/Ranluxpp.h>

#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmParametersInit.hh>

#define NOMSC
#define NOFLUCTUATION

#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
// Pull in implementation.
#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmElectronInteractionUMSC.icc> // Needed to make a debug build succeed.
#include <G4HepEmPositronInteractionAnnihilation.icc>

__constant__ __device__ struct G4HepEmParameters g4HepEmPars;
__constant__ __device__ struct G4HepEmData g4HepEmData;

struct G4HepEmState {
  G4HepEmData data;
  G4HepEmParameters parameters;
};

static G4HepEmState *InitG4HepEm()
{
  G4HepEmState *state = new G4HepEmState;
  InitG4HepEmData(&state->data);
  InitHepEmParameters(&state->parameters);

  InitMaterialAndCoupleData(&state->data, &state->parameters);

  InitElectronData(&state->data, &state->parameters, true);
  InitElectronData(&state->data, &state->parameters, false);

  G4HepEmMatCutData *cutData = state->data.fTheMatCutData;
  printf("fNumG4MatCuts = %d, fNumMatCutData = %d\n", cutData->fNumG4MatCuts, cutData->fNumMatCutData);

  // Copy to GPU.
  CopyG4HepEmDataToGPU(&state->data);
  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmPars, &state->parameters, sizeof(G4HepEmParameters)));

  // Create G4HepEmData with the device pointers.
  G4HepEmData dataOnDevice;
  dataOnDevice.fTheMatCutData   = state->data.fTheMatCutData_gpu;
  dataOnDevice.fTheMaterialData = state->data.fTheMaterialData_gpu;
  dataOnDevice.fTheElementData  = state->data.fTheElementData_gpu;
  dataOnDevice.fTheElectronData = state->data.fTheElectronData_gpu;
  dataOnDevice.fThePositronData = state->data.fThePositronData_gpu;
  dataOnDevice.fTheSBTableData  = state->data.fTheSBTableData_gpu;
  // The other pointers should never be used.
  dataOnDevice.fTheMatCutData_gpu   = nullptr;
  dataOnDevice.fTheMaterialData_gpu = nullptr;
  dataOnDevice.fTheElementData_gpu  = nullptr;
  dataOnDevice.fTheElectronData_gpu = nullptr;
  dataOnDevice.fThePositronData_gpu = nullptr;
  dataOnDevice.fTheSBTableData_gpu  = nullptr;

  COPCORE_CUDA_CHECK(cudaMemcpyToSymbol(g4HepEmData, &dataOnDevice, sizeof(G4HepEmData)));

  return state;
}

static void FreeG4HepEm(G4HepEmState *state)
{
  FreeG4HepEmData(&state->data);
  delete state;
}

class RanluxppDoubleEngine : public G4HepEmRandomEngine {
  // Wrapper functions to call into CLHEP::HepRandomEngine.
  static __host__ __device__ double flatWrapper(void *object) {
    return ((RanluxppDouble*)object)->Rndm();
  }
  static __host__ __device__ void flatArrayWrapper(void *object, const int size, double* vect) {
    for (int i = 0; i < size; i++) {
      vect[i] = ((RanluxppDouble*)object)->Rndm();
    }
  }

public:
  __host__ __device__
  RanluxppDoubleEngine(RanluxppDouble* engine)
    : G4HepEmRandomEngine(/*object=*/engine, &flatWrapper, &flatArrayWrapper) {}
};

__global__ void TransportParticle()
{
  RanluxppDouble r;
  RanluxppDoubleEngine rnge(&r);

  // Init a track.
  G4HepEmElectronTrack elTrack;
  // To simplify copy&paste...
  G4HepEmElectronTrack *theElTrack = &elTrack;
  G4HepEmTrack *theTrack           = elTrack.GetTrack();
  theTrack->SetEKin(100 * copcore::units::GeV);
  theTrack->SetMCIndex(1);
  const bool isElectron = true;
  printf("Starting with %fMeV\n", theTrack->GetEKin());

  for (int i = 0; i < 200; i++) {
    printf("-----------------------------------------\n");
    // Sample the `number-of-interaction-left`.
    for (int ip = 0; ip < 3; ++ip) {
      if (theTrack->GetNumIALeft(ip) <= 0.) {
        theTrack->SetNumIALeft(-std::log(r.Rndm()), ip);
      }
    }

    G4HepEmElectronManager::HowFar(&g4HepEmData, &g4HepEmPars, theElTrack, nullptr);
    printf("sampled process: %d, particle travels %fmm\n", theTrack->GetWinnerProcessIndex(),
           theTrack->GetGStepLength());

    const int iDProc = theTrack->GetWinnerProcessIndex();
    bool stopped     = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, theElTrack, nullptr);
    printf("energy after continuous process: %fMeV\n", theTrack->GetEKin());
    if (stopped) {
      // call annihilation for e+ !!!
      if (!isElectron) {
        // FIXME !!!
        // PerformPositronAnnihilation(tlData, true);
      }
      return;
    } else if (iDProc < 0) {
      // No discrete process or on boundary.
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    theTrack->SetNumIALeft(-1.0, iDProc);

    // Check if a delta interaction happens instead of the real discrete process.
    if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, r.Rndm())) {
      printf("delta interaction happened!\n");
      continue;
    }

    // TODO: Perform the discrete part of the winner interaction.
    const int theMCIndx     = theTrack->GetMCIndex();
    const double theEkin    = theTrack->GetEKin();
    const double theLogEkin = theTrack->GetLogEKin();
    const double theElCut   = g4HepEmData.fTheMatCutData->fMatCutData[theMCIndx].fSecElProdCutE;
    switch (iDProc) {
    case 0: {
      // invoke ioni (for e-/e+):
      // PerformElectronIoni(tlData, hepEmData, isElectron);
      const double deltaEkin = (isElectron)
                                   ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, theEkin, &rnge)
                                   : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, theEkin, &rnge);
      theTrack->SetEKin(theEkin - deltaEkin);
      break;
    }
    case 1: // invoke brem (for e-/e+): either SB- or Rel-Brem
      if (theEkin < g4HepEmPars.fElectronBremModelLim) {
        // PerformElectronBremSB(tlData, hepEmData, isElectron);
        double deltaEkin = G4HepEmElectronInteractionBrem::SampleETransferSB(&g4HepEmData, theEkin, theLogEkin,
                                                                             theMCIndx, &rnge, isElectron);
        theTrack->SetEKin(theEkin - deltaEkin);
      } else {
        // PerformElectronBremRB(tlData, hepEmData);
        double deltaEkin = G4HepEmElectronInteractionBrem::SampleETransferRB(&g4HepEmData, theEkin, theLogEkin,
                                                                             theMCIndx, &rnge, isElectron);
        theTrack->SetEKin(theEkin - deltaEkin);
      }
      break;
    case 2: // invoke annihilation (in-flight) for e+
      // PerformPositronAnnihilation(tlData, false);
      break;
    }
    printf("energy after discrete process:   %fMeV\n", theTrack->GetEKin());
  }
}

void example5()
{
  G4HepEmState *state = InitG4HepEm();

  printf("Launching particle transport on GPU\n");
  printf("-----------------------------------------\n");

  TransportParticle<<<1, 1>>>();
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  FreeG4HepEm(state);
}
