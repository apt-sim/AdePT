// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/integration/G4EmStandardPhysics_AdePT.hh>

#include <AdePT/core/AdePTConfiguration.hh>
#include <AdePT/integration/AdePTTrackingManager.hh>

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"

G4EmStandardPhysics_AdePT::G4EmStandardPhysics_AdePT(G4int ver, const G4String &name)
    : G4EmStandardPhysics(ver, name), fAdePTConfiguration(new AdePTConfiguration())
{
}

G4EmStandardPhysics_AdePT::~G4EmStandardPhysics_AdePT()
{
  delete fAdePTConfiguration;
  // Deleting the tracking manager can crash on some Geant4 releases.
  // Keep Geant4-owned cleanup behavior for compatibility.
}

void G4EmStandardPhysics_AdePT::ConstructProcess()
{
  // First register the standard Geant4 EM processes for this constructor.
  G4EmStandardPhysics::ConstructProcess();

  if (fTrackingManager == nullptr) {
    fTrackingManager = new AdePTTrackingManager(fAdePTConfiguration, /*verbosity=*/0);
  }

  auto g4hepemconfig = fTrackingManager->GetG4HepEmConfig();
  g4hepemconfig->SetMultipleStepsInMSCWithTransportation(
      fAdePTConfiguration->GetMultipleStepsInMSCWithTransportation());
  g4hepemconfig->SetEnergyLossFluctuation(fAdePTConfiguration->GetEnergyLossFluctuation());

  for (const auto &regionName : fAdePTConfiguration->GetWDTRegionNames()) {
    g4hepemconfig->SetWoodcockTrackingRegion(regionName);
  }
  g4hepemconfig->SetWDTEnergyLimit(fAdePTConfiguration->GetWDTKineticEnergyLimit());

  G4Electron::Definition()->SetTrackingManager(fTrackingManager);
  G4Positron::Definition()->SetTrackingManager(fTrackingManager);
  G4Gamma::Definition()->SetTrackingManager(fTrackingManager);
}
