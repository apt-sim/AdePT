// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <G4NistManager.hh>
#include <G4Material.hh>

#include <G4Box.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>

#include <G4ParticleTable.hh>
#include <G4Electron.hh>
#include <G4Positron.hh>
#include <G4Gamma.hh>
#include <G4Proton.hh>

#include <G4ProductionCuts.hh>
#include <G4Region.hh>
#include <G4ProductionCutsTable.hh>

#include <G4UnitsTable.hh>
#include <G4SystemOfUnits.hh>

#include <G4HepEmData.hh>
#include <G4HepEmElectronInit.hh>
#include <G4HepEmMatCutData.hh>
#include <G4HepEmMaterialInit.hh>
#include <G4HepEmParameters.hh>
#include <G4HepEmParametersInit.hh>

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
#include <G4HepEmPositronInteractionAnnihilation.icc>

#include <CopCore/Global.h>
#include <CopCore/Ranluxpp.h>

static void InitGeant4()
{
  // --- Create materials.
  G4Material *galactic = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
  G4Material *silicon  = G4NistManager::Instance()->FindOrBuildMaterial("G4_Si");
  //
  // --- Define a world.
  G4double worldDim         = 1 * m;
  G4Box *worldBox           = new G4Box("world", worldDim, worldDim, worldDim);
  G4LogicalVolume *worldLog = new G4LogicalVolume(worldBox, galactic, "world");
  G4PVPlacement *world      = new G4PVPlacement(nullptr, {}, worldLog, "world", nullptr, false, 0);
  // --- Define a box.
  G4double boxDim             = 0.5 * m;
  G4double boxPos             = 0.5 * boxDim;
  G4Box *siliconBox           = new G4Box("silicon", boxDim, boxDim, boxDim);
  G4LogicalVolume *siliconLog = new G4LogicalVolume(siliconBox, silicon, "silicon");
  new G4PVPlacement(nullptr, {boxPos, boxPos, boxPos}, siliconLog, "silicon", worldLog, false, 0);
  //
  // --- Create particles that have secondary production threshold.
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable *partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();
  //
  // --- Create production - cuts object and set the secondary production threshold.
  G4ProductionCuts *productionCuts = new G4ProductionCuts();
  constexpr G4double ProductionCut = 1 * mm;
  productionCuts->SetProductionCut(ProductionCut);
  //
  // --- Register a region for the world.
  G4Region *reg = new G4Region("default");
  reg->AddRootLogicalVolume(worldLog);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(productionCuts);
  //
  // --- Update the couple tables.
  G4ProductionCutsTable *theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(world);
}

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
  G4cout << "fNumG4MatCuts = " << cutData->fNumG4MatCuts << ", fNumMatCutData = " << cutData->fNumMatCutData << G4endl;

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
  theTrack->SetEKin(100 * GeV);
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

    G4HepEmElectronManager::HowFar(&g4HepEmData, &g4HepEmPars, theElTrack);
    printf("sampled process: %d, particle travels %fmm\n", theTrack->GetWinnerProcessIndex(),
           theTrack->GetGStepLength());

    const int iDProc = theTrack->GetWinnerProcessIndex();
    bool stopped     = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, theElTrack);
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

int main()
{
  InitGeant4();
  G4HepEmState *state = InitG4HepEm();

  printf("Launching particle transport on GPU\n");
  printf("-----------------------------------------\n");

  TransportParticle<<<1, 1>>>();
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());

  FreeG4HepEm(state);

  return 0;
}
