// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include "PhysListAdePT.hh"

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

#include "DetectorConstruction.hh"
#include "SensitiveDetector.hh"

PhysListAdePT::PhysListAdePT(DetectorConstruction *aDetector, const G4String &name) : fDetector(aDetector), G4VPhysicsConstructor(name)
{
  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(1);

  // Range factor: (can be set from the G4 macro)
  param->SetMscRangeFactor(0.04);
  //

  SetPhysicsType(bElectromagnetic);
}

PhysListAdePT::~PhysListAdePT() {}

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

  auto caloSD = dynamic_cast<SensitiveDetector*>(G4SDManager::GetSDMpointer()->FindSensitiveDetector("AdePTDetector"));
  fTrackingManager->SetSensitiveVolumes(&(caloSD->fSensitive_volume_index));
  fTrackingManager->SetScoringMap(caloSD->fScoringMap);
  fTrackingManager->SetVerbosity(0);
  fTrackingManager->SetBufferThreshold(fDetector->GetBufferThreshold());
  fTrackingManager->SetTrackSlots(fDetector->GetTrackSlots());

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
}
