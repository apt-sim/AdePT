// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdePTPhysics.hh>
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

#include "G4GeometryManager.hh"
#include "G4RunManager.hh"

AdePTPhysics::AdePTPhysics(const G4String &name) : G4VPhysicsConstructor(name)
{
  fAdePTConfiguration = new AdePTConfiguration();

  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(1);

  // Range factor: (can be set from the G4 macro)
  // param->SetMscRangeFactor(0.04);
  //

  SetPhysicsType(bUnknown);
}

AdePTPhysics::~AdePTPhysics() 
{
  delete fAdePTConfiguration;
  delete fTrackingManager;
}

void AdePTPhysics::ConstructProcess()
{
  G4PhysicsListHelper *ph = G4PhysicsListHelper::GetPhysicsListHelper();

  // Register custom tracking manager for e-/e+ and gammas.
  fTrackingManager = new AdePTTrackingManager();
  G4Electron::Definition()->SetTrackingManager(fTrackingManager);
  G4Positron::Definition()->SetTrackingManager(fTrackingManager);
  G4Gamma::Definition()->SetTrackingManager(fTrackingManager);

  // Setup tracking manager
  fTrackingManager->SetVerbosity(0);
  
  // Create one instance of AdePTTransport per thread and set it up according to the user configuration
  AdePTTransport *aAdePTTransport = new AdePTTransport();
  aAdePTTransport->SetDebugLevel(0);
  aAdePTTransport->SetBufferThreshold(fAdePTConfiguration->GetTransportBufferThreshold());
  aAdePTTransport->SetMaxBatch(2 * fAdePTConfiguration->GetTransportBufferThreshold());
  aAdePTTransport->SetTrackInAllRegions(fAdePTConfiguration->GetTrackInAllRegions());
  aAdePTTransport->SetGPURegionNames(fAdePTConfiguration->GetGPURegionNames());

  // Check if this is a sequential run
  G4RunManager::RMType rmType = G4RunManager::GetRunManager()->GetRunManagerType();
  bool sequential             = (rmType == G4RunManager::sequentialRM);

  // One thread initializes common elements
  auto tid = G4Threading::G4GetThreadId();
  if (tid < 0) {
    // Load the VecGeom world in memory
    AdePTGeant4Integration::CreateVecGeomWorld(fAdePTConfiguration->GetVecGeomGDML());
    
    // Track and Hit buffer capacities on GPU are split among threads
    int num_threads = G4RunManager::GetRunManager()->GetNumberOfThreads();
    int track_capacity    = 1024 * 1024 * fAdePTConfiguration->GetMillionsOfTrackSlots() / num_threads;
    G4cout << "AdePT Allocated track capacity: " << track_capacity << " tracks" << G4endl;
    AdePTTransport::SetTrackCapacity(track_capacity);
    int hit_buffer_capacity = 1024 * 1024 * fAdePTConfiguration->GetMillionsOfHitSlots() / num_threads;
    G4cout << "AdePT Allocated hit buffer capacity: " << hit_buffer_capacity << " slots" << G4endl;
    AdePTTransport::SetHitBufferCapacity(hit_buffer_capacity);

    // Initialize common data:
    // G4HepEM, Upload VecGeom geometry to GPU, Geometry check, Create volume auxiliary data
    aAdePTTransport->Initialize(true /*common_data*/);
    if (sequential) 
    {
      // Initialize per-thread data (When in sequential mode)
      aAdePTTransport->Initialize();
    }
  } else {
    // Initialize per-thread data
    aAdePTTransport->Initialize();
  }
  
  // Give the custom tracking manager a pointer to the AdePTTransport instance
  fTrackingManager->SetAdePTTransport(aAdePTTransport);

  // Translate Region names to actual G4 Regions and give them to the custom tracking manager


}
