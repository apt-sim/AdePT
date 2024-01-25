// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/PhysListAdePT.hh>
#include <AdePT/integration/AdePTGeant4Integration.hh>

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4PhysicsListHelper.hh"

#include "G4ComptonScattering.hh"
//#include "G4KleinNishinaModel.hh"  // by defult in G4ComptonScattering

#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4LivermorePhotoElectricModel.hh"
//#include "G4RayleighScattering.hh"

#include "G4eMultipleScattering.hh"
#include "G4GoudsmitSaundersonMscModel.hh"
#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4EmParameters.hh"
#include "G4MscStepLimitType.hh"

#include "G4BuilderType.hh"
#include "G4LossTableManager.hh"
//#include "G4UAtomicDeexcitation.hh"

#include "G4SystemOfUnits.hh"

// from G4EmStandardPhysics
#include "G4GenericIon.hh"
#include "G4EmModelActivator.hh"
#include "G4EmBuilder.hh"
#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4ionIonisation.hh"
#include "G4NuclearStopping.hh"
#include "G4SDManager.hh"

#include "G4RunManager.hh"

PhysListAdePT::PhysListAdePT(const G4String &name) : G4VPhysicsConstructor(name)
{
  fAdePTConfiguration = new AdePTConfiguration();

  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(1);

  // Range factor: (can be set from the G4 macro)
  param->SetMscRangeFactor(0.04);
  //

  SetPhysicsType(bElectromagnetic);
}

PhysListAdePT::~PhysListAdePT() 
{
  delete fAdePTConfiguration;
  delete fTrackingManager;
}

void PhysListAdePT::ConstructProcess()
{
  G4PhysicsListHelper *ph = G4PhysicsListHelper::GetPhysicsListHelper();

  // from G4EmStandardPhysics
  G4EmBuilder::PrepareEMPhysics();

  G4EmParameters* param = G4EmParameters::Instance();

  // processes used by several particles
  G4hMultipleScattering* hmsc = new G4hMultipleScattering("ionmsc");

  // nuclear stopping is enabled if th eenergy limit above zero
  G4double nielEnergyLimit = param->MaxNIELEnergy();
  G4NuclearStopping* pnuc = nullptr;
  if(nielEnergyLimit > 0.0) {
    pnuc = new G4NuclearStopping();
    pnuc->SetMaxKinEnergy(nielEnergyLimit);
  }
  // end of G4EmStandardPhysics

  // Register custom tracking manager for e-/e+ and gammas.
  fTrackingManager = new AdePTTrackingManager();
  G4Electron::Definition()->SetTrackingManager(fTrackingManager);
  G4Positron::Definition()->SetTrackingManager(fTrackingManager);
  G4Gamma::Definition()->SetTrackingManager(fTrackingManager);

  // Setup tracking manager
  fTrackingManager->SetVerbosity(0);

  // from G4EmStandardPhysics

  // generic ion
  G4ParticleDefinition* particle = G4GenericIon::GenericIon();
  G4ionIonisation* ionIoni = new G4ionIonisation();
  ph->RegisterProcess(hmsc, particle);
  ph->RegisterProcess(ionIoni, particle);
  if(nullptr != pnuc) { ph->RegisterProcess(pnuc, particle); }

  // muons, hadrons ions
  G4EmBuilder::ConstructCharged(hmsc, pnuc);

  // extra configuration
  G4EmModelActivator mact(GetPhysicsName());

  //end of G4EmStandardPhysics

  auto caloSD = dynamic_cast<G4VSensitiveDetector*>(G4SDManager::GetSDMpointer()->FindSensitiveDetector("AdePTDetector"));
  
  AdePTTransport *adept = new AdePTTransport();

  adept->SetDebugLevel(0);
  adept->SetBufferThreshold(fAdePTConfiguration->GetTransportBufferThreshold());
  adept->SetMaxBatch(2 * fAdePTConfiguration->GetTransportBufferThreshold());

  G4RunManager::RMType rmType = G4RunManager::GetRunManager()->GetRunManagerType();
  bool sequential             = (rmType == G4RunManager::sequentialRM);

  //adept->SetSensitiveVolumes(&(caloSD->fSensitiveLogicalVolumes));
  adept->SetRegion(nullptr);

  auto tid = G4Threading::G4GetThreadId();
  if (tid < 0) {
    //adept->CreateVecGeomWorld();
    AdePTGeant4Integration::CreateVecGeomWorld();
    // This is supposed to set the max batching for Adept to allocate properly the memory
    int num_threads = G4RunManager::GetRunManager()->GetNumberOfThreads();
    int track_capacity    = 1024 * 1024 * fAdePTConfiguration->GetMillionsOfTrackSlots() / num_threads;
    G4cout << "AdePT Allocated track capacity: " << track_capacity << " tracks" << G4endl;
    AdePTTransport::SetTrackCapacity(track_capacity);
    int hit_buffer_capacity = 1024 * 1024 * fAdePTConfiguration->GetMillionsOfHitSlots() / num_threads;
    G4cout << "AdePT Allocated hit buffer capacity: " << hit_buffer_capacity << " slots" << G4endl;
    AdePTTransport::SetHitBufferCapacity(hit_buffer_capacity);
    adept->Initialize(true /*common_data*/);
    if (sequential) 
    {
      adept->Initialize();
    }
  } else {
    adept->Initialize();
  }
  
  fTrackingManager->SetAdePTTransport(adept);
}
