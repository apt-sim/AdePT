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

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StoppingPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4HadronPhysicsFTFP_BERT.hh"

#include <AdePT/integration/HepEMPhysics.hh>
#include <AdePT/integration/AdePTPhysics.hh>
#include "FTFP_BERT_AdePT.hh"

FTFP_BERT_AdePT::FTFP_BERT_AdePT(G4int ver)
{
  // default cut value  (1.0mm)
  // defaultCutValue = 1.0*CLHEP::mm;
  G4cout << "<<< Geant4 Physics List simulation engine: FTFP_BERT_AdePT" << G4endl;
  G4cout << G4endl;
  defaultCutValue = 0.7 * CLHEP::mm;
  SetVerboseLevel(ver);

  // EM Physics

  // Register the EM physics to use for tracking on CPU
  // RegisterPhysics(new G4EmStandardPhysics());
  RegisterPhysics(new HepEMPhysics());

  // Register the AdePT physics
  RegisterPhysics(new AdePTPhysics());

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
