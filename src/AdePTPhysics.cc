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

  // Apply Woodcock tracking of photons in the EMEC and EMB
  // g4hepemconfig->SetWoodcockTrackingRegion("caloregion");
  g4hepemconfig->SetWoodcockTrackingRegion("HGCalRegion");
  g4hepemconfig->SetWDTEnergyLimit(0.5); // set to 500 keV instead of 200
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer2");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer1");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer2");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer3");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer4");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer5");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer6");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer7");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer8");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer9");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer10");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer11");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer12");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer13");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer14");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer15");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer16");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer17");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer18");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer19");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer20");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer21");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer22");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer23");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer24");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer25");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer26");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer27");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer28");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer29");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer30");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer31");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer32");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer33");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer34");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer35");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer36");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer37");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer38");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer39");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer40");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer41");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer42");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer43");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer44");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer45");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer46");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer47");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer48");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer49");
  // g4hepemconfig->SetWoodcockTrackingRegion("Layer50");

  // g4hepemconfig->SetWoodcockTrackingRegion("ZDCRegion");
  // g4hepemconfig->SetWoodcockTrackingRegion("MuonSensitive_RPC");
  // g4hepemconfig->SetWoodcockTrackingRegion("APDRegion");

  // DefaultRegionForTheWorld | DefaultRegionForParallelWorld | ZDCRegion | TrackerDeadRegion | TrackerSensRegion |
  // MuonChamber | MuonSensitive_RPC | TrackerPixelSensRegion | TrackerPixelDeadRegion | MuonIron | Muon |
  // MuonSensitive_DT-CSC | HcalRegion | QuadRegion | PreshowerSensRegion | PreshowerRegion | EcalRegion | APDRegion |
  // InterimRegion | CastorRegion | BeamPipeVacuum | BeamPipe | BeamPipeOutside |

  G4Electron::Definition()->SetTrackingManager(fTrackingManager);
  G4Positron::Definition()->SetTrackingManager(fTrackingManager);
  G4Gamma::Definition()->SetTrackingManager(fTrackingManager);
}
