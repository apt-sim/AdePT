// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <iomanip>

#include "globals.hh"
#include "G4ios.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4ParticleTypes.hh"
#include "G4ParticleTable.hh"

#include "G4Material.hh"
#include "G4MaterialTable.hh"

#include "PhysListHepEm.hh"
#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"

#include "G4HadronPhysicsFTFP_BERT.hh"
#include "FTFP_BERT_HepEm.hh"

FTFP_BERT_HepEm::FTFP_BERT_HepEm(G4int ver)
{
  // default cut value  (1.0mm)
  // defaultCutValue = 1.0*CLHEP::mm;
  G4cout << "<<< Geant4 Physics List simulation engine: FTFP_BERT_HepEm" << G4endl;
  G4cout << G4endl;
  defaultCutValue = 0.7 * CLHEP::mm;
  SetVerboseLevel(ver);

  // EM Physics
  //  RegisterPhysics( new G4EmStandardPhysics(ver));
  RegisterPhysics(new PhysListHepEm);

  // Synchroton Radiation & GN Physics
  // comenting out to remove gamma- and lepto-nuclear processes
  // RegisterPhysics( new G4EmExtraPhysics(ver) );

  // Decays
  RegisterPhysics(new G4DecayPhysics(ver));

  // Hadron Elastic scattering
  RegisterPhysics(new G4HadronElasticPhysics(ver));

  // Hadron Physics
  RegisterPhysics(new G4HadronPhysicsFTFP_BERT(ver));

  // Stopping Physics
  RegisterPhysics(new G4StoppingPhysics(ver));

  // Ion Physics
  RegisterPhysics(new G4IonPhysics(ver));

  // Neutron tracking cut
  RegisterPhysics(new G4NeutronTrackingCut(ver));
}
