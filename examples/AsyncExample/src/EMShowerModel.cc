// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "EMShowerMessenger.hh"

#include <G4RunManager.hh>
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4FastHit.hh"
#include "Randomize.hh"
#include "G4FastSimHitMaker.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"

#include "EMShowerModel.hh"
#include "G4GlobalFastSimulationManager.hh"

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>

#include <G4MaterialCutsCouple.hh>
#include <G4ProductionCutsTable.hh>

EMShowerModel::EMShowerModel(G4String aModelName, G4Region *aEnvelope, std::shared_ptr<AdeptIntegration> adept)
    : G4VFastSimulationModel{aModelName, aEnvelope}, fMessenger{new EMShowerMessenger(this)}, fAdept{adept}
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EMShowerModel::EMShowerModel(G4String aModelName)
    : G4VFastSimulationModel(aModelName), fMessenger(new EMShowerMessenger(this))
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EMShowerModel::~EMShowerModel() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool EMShowerModel::IsApplicable(const G4ParticleDefinition &aParticleType)
{
  return fAdept && (&aParticleType == G4Electron::ElectronDefinition() ||
                    &aParticleType == G4Positron::PositronDefinition() || &aParticleType == G4Gamma::GammaDefinition());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool EMShowerModel::ModelTrigger(const G4FastTrack & /*aFastTrack*/)
{

  // The model is invoked for e/e-/gamma, so this has to return true
  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EMShowerModel::DoIt(const G4FastTrack &aFastTrack, G4FastStep &aFastStep)
{
  auto g4track           = aFastTrack.GetPrimaryTrack();
  auto particlePosition  = g4track->GetPosition();
  auto particleDirection = g4track->GetMomentumDirection();

#ifndef NDEBUG
  G4VSolid const *solid = aFastTrack.GetEnvelopeSolid();
  const auto localPos   = aFastTrack.GetPrimaryTrackLocalPosition();
  const auto localDir   = aFastTrack.GetPrimaryTrackLocalDirection();
  const auto dist       = solid->DistanceToOut(localPos, localDir);
  // if (dist < millimeter) std::cout << "DistanceToOut: " << std::setw(9) << dist / millimeter << "\n";
  if (dist < micrometer) {
    G4cout << "Error: AdePT will reject a particle that's immediately leaving the GPU region.\n";
    G4cout << "Rejected pos: " << aFastTrack.GetPrimaryTrack()->GetPosition() << "\n";
    G4cout << "Distance: " << solid->DistanceToOut(localPos, localDir) << "\n";
    G4cout << *solid << "\n";
    return;
  }

  const G4ThreeVector pos{-282.494219, -1126.731331, -3156.200000};
  if ((particlePosition - pos).mag() < 1.E-5) {
    G4cout << "TAG: The solid close to " << particlePosition << " is:\n";
    G4cout << *solid << "\n";
    G4cout << "LVol=" << aFastTrack.GetEnvelopeLogicalVolume()->GetName()
           << " NDaugher=" << aFastTrack.GetEnvelopeLogicalVolume()->GetNoDaughters() << "\n";
    G4cout << "DistanceToOut=" << solid->DistanceToOut(localPos, localDir) / millimeter << G4endl;
  }

  const G4ThreeVector posKickOut{129.079191, 1231.252437, 888.105959};
  if ((particlePosition - posKickOut).mag() < 1.E-4) {
    G4cout << "TAG: The solid close to the kick-out position " << particlePosition << " is:\n";
    G4cout << *solid << "\n";
    G4cout << "LVol=" << aFastTrack.GetEnvelopeLogicalVolume()->GetName()
           << " NDaugher=" << aFastTrack.GetEnvelopeLogicalVolume()->GetNoDaughters() << "\n";
    G4cout << "DistanceToOut=" << solid->DistanceToOut(localPos, localDir) / millimeter << G4endl;
  }
#endif

  // Remove particle from further processing by G4
  aFastStep.KillPrimaryTrack();
  aFastStep.SetPrimaryTrackPathLength(0.0);
  G4double energy = aFastTrack.GetPrimaryTrack()->GetKineticEnergy();
  // No need to create any deposit, it will be handled by this model (and
  // G4FastSimHitMaker that will call the sensitive detector)
  aFastStep.SetTotalEnergyDeposited(0);

  auto pdg = aFastTrack.GetPrimaryTrack()->GetParticleDefinition()->GetPDGEncoding();
  const auto thread = G4Threading::G4GetThreadId();
  const auto event  = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
  if (event != fLastEventId) {
    fLastEventId  = event;
    fTrackCounter = 0;
    fCycleNumber  = 0;
  }

  fAdept->AddTrack(thread, event, fCycleNumber, fTrackCounter++, pdg, energy, particlePosition[0], particlePosition[1],
                   particlePosition[2], particleDirection[0], particleDirection[1], particleDirection[2]);
}

void EMShowerModel::Flush()
{
  const auto threadId = G4Threading::G4GetThreadId();
  const auto event    = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
  if (fVerbosity > 0)
    G4cout << "Waiting for AdePT to finish the transport for thread " << threadId << " event " << event << " cycle "
           << fCycleNumber << G4endl;

  fAdept->Flush(threadId, event, fCycleNumber);
  fCycleNumber++;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EMShowerModel::Print() const
{
  G4cout << "EMShowerModel (AdePT)=" << fAdept << G4endl;
}
