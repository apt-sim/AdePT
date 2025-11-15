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

AdePTPhysics::AdePTPhysics(int ver, const G4String &name) : G4VPhysicsConstructor(name)
{
  fAdePTConfiguration = new AdePTConfiguration();

  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(1);

  // Range factor: (can be set from the G4 macro)
  // param->SetMscRangeFactor(0.04); // 0.04 is the default set by SetDefaults

  SetPhysicsType(bUnknown);
}

AdePTPhysics::~AdePTPhysics()
{
  delete fAdePTConfiguration;
  // the delete below causes a crash with G4.10.7
  // delete fTrackingManager;
}

void AdePTPhysics::ConstructProcess()
{
  // Register custom tracking manager for e-/e+ and gammas.
  fTrackingManager = new AdePTTrackingManager(fAdePTConfiguration, /*verbosity=*/0);

  auto g4hepemconfig = fTrackingManager->GetG4HepEmConfig();
  g4hepemconfig->SetMultipleStepsInMSCWithTransportation(
      fAdePTConfiguration->GetMultipleStepsInMSCWithTransportation());
  g4hepemconfig->SetEnergyLossFluctuation(fAdePTConfiguration->GetEnergyLossFluctuation());

  // Loop over all configured Woodcock regions and register them
  for (const auto &regionName : fAdePTConfiguration->GetWDTRegionNames()) {
    g4hepemconfig->SetWoodcockTrackingRegion(regionName);
  }
  // set Woodcock tracking energy limit
  g4hepemconfig->SetWDTEnergyLimit(fAdePTConfiguration->GetWDTKineticEnergyLimit());

  G4Electron::Definition()->SetTrackingManager(fTrackingManager);
  G4Positron::Definition()->SetTrackingManager(fTrackingManager);
  G4Gamma::Definition()->SetTrackingManager(fTrackingManager);
}
