// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "HepMC3G4Interface.hh"

#include "G4RunManager.hh"
#include "G4LorentzVector.hh"
#include "G4Event.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4TransportationManager.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4Interface::HepMC3G4Interface() : hepmcEvent(nullptr) {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
HepMC3G4Interface::~HepMC3G4Interface()
{
  delete hepmcEvent;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
G4bool HepMC3G4Interface::CheckVertexInsideWorld(const G4ThreeVector &pos) const
{
  G4Navigator *navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();

  G4VPhysicalVolume *world = navigator->GetWorldVolume();
  G4VSolid *solid          = world->GetLogicalVolume()->GetSolid();
  EInside qinside          = solid->Inside(pos);

  if (qinside != kInside)
    return false;
  else
    return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void HepMC3G4Interface::HepMC2G4(const HepMC3::GenEvent *hepmcevt, G4Event *g4event)
{
  for (auto vitr : hepmcevt->vertices()) { // loop for vertex ...

    // real vertex?
    G4bool qvtx = false;
    for (auto pitr : vitr->particles_out()) {

      if (!pitr->end_vertex() && pitr->status() == 1) {
        qvtx = true;
        break;
      }
    }
    if (!qvtx) continue;

    // check world boundary
    HepMC3::FourVector pos = vitr->position();
    G4LorentzVector xvtx(pos.x(), pos.y(), pos.z(), pos.t());
    if (!CheckVertexInsideWorld(xvtx.vect() * mm)) continue;

    // create G4PrimaryVertex and associated G4PrimaryParticles
    G4PrimaryVertex *g4vtx = new G4PrimaryVertex(xvtx.x() * mm, xvtx.y() * mm, xvtx.z() * mm, xvtx.t() * mm / c_light);

    for (auto vpitr : vitr->particles_out()) {

      if (vpitr->status() != 1) continue;

      G4int pdgcode = vpitr->pdg_id();
      pos           = vpitr->momentum();
      G4LorentzVector p(pos.px(), pos.py(), pos.pz(), pos.e());
      G4PrimaryParticle *g4prim = new G4PrimaryParticle(pdgcode, p.x() * GeV, p.y() * GeV, p.z() * GeV);
      g4vtx->SetPrimary(g4prim);
    }
    g4event->AddPrimaryVertex(g4vtx);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void HepMC3G4Interface::GeneratePrimaryVertex(G4Event *anEvent)
{
  // generate next event
  hepmcEvent = GenerateHepMCEvent(anEvent->GetEventID());
  if (!hepmcEvent) {
    G4cout << "HepMCInterface: no generated particles. run terminated..." << G4endl;
    G4RunManager::GetRunManager()->AbortRun();
    return;
  }
  HepMC2G4(hepmcEvent, anEvent);
}
