// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/AdePTPhysics.hh>

#include <AdePT/integration/AdePTGeant4Integration.hh>
#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/core/AdePTTransportInterface.hh>
#include <AdePT/integration/AdePTTrackingManager.hh>

#include "G4ParticleDefinition.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4EmParameters.hh"
#include "G4BuilderType.hh"

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
  // the delete below causes a crash with G4.10.7
  //delete fTrackingManager;
}

void AdePTPhysics::ConstructProcess()
{
  // Register custom tracking manager for e-/e+ and gammas.
  fTrackingManager = new AdePTTrackingManager();
  G4Electron::Definition()->SetTrackingManager(fTrackingManager);
  G4Positron::Definition()->SetTrackingManager(fTrackingManager);
  G4Gamma::Definition()->SetTrackingManager(fTrackingManager);

  // Setup tracking manager
  fTrackingManager->SetVerbosity(0);
  fTrackingManager->SetAdePTConfiguration(fAdePTConfiguration);
}
