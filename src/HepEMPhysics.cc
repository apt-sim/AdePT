// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/HepEMPhysics.hh>

// include the G4HepEmProcess from the G4HepEm lib.
// #include "G4HepEmProcess.hh"

// Using the tracking manager approach
#include "G4HepEmTrackingManager.hh"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4PhysicsListHelper.hh"

#include "G4ComptonScattering.hh"
// #include "G4KleinNishinaModel.hh"  // by defult in G4ComptonScattering

#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4LivermorePhotoElectricModel.hh"
// #include "G4RayleighScattering.hh"

#include "G4eMultipleScattering.hh"
#include "G4GoudsmitSaundersonMscModel.hh"
#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4EmParameters.hh"
#include "G4MscStepLimitType.hh"

#include "G4BuilderType.hh"
#include "G4LossTableManager.hh"
// #include "G4UAtomicDeexcitation.hh"

#include "G4SystemOfUnits.hh"

// from G4EmStandardPhysics
#include "G4GenericIon.hh"
#include "G4EmModelActivator.hh"
#include "G4EmBuilder.hh"
#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4ionIonisation.hh"
#include "G4NuclearStopping.hh"

HepEMPhysics::HepEMPhysics(int ver, const G4String &name) : G4VPhysicsConstructor(name)
{
  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(1);

  // Range factor: (can be set from the G4 macro)
  // param->SetMscRangeFactor(0.04); // 0.04 is the default set by SetDefaults
}

HepEMPhysics::~HepEMPhysics() {}

void HepEMPhysics::ConstructProcess()
{
  // Register custom tracking manager for e-/e+ and gammas.
  auto *trackingManager = new G4HepEmTrackingManager();
  G4Electron::Definition()->SetTrackingManager(trackingManager);
  G4Positron::Definition()->SetTrackingManager(trackingManager);
  G4Gamma::Definition()->SetTrackingManager(trackingManager);

  // end of HepEMPhysics
}
