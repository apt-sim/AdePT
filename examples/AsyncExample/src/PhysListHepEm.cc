// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include "PhysListHepEm.hh"

// include the G4HepEmProcess from the G4HepEm lib.
#include "G4HepEmProcess.hh"

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



PhysListHepEm::PhysListHepEm(const G4String &name) : G4VPhysicsConstructor(name)
{
  G4EmParameters *param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(1);

  // Range factor: (can be set from the G4 macro)
  param->SetMscRangeFactor(0.04);
  //

  SetPhysicsType(bElectromagnetic);
}

PhysListHepEm::~PhysListHepEm() {}

void PhysListHepEm::ConstructProcess()
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

  // creae the only one G4HepEm process that will be assigned to e-/e+ and gamma
  G4HepEmProcess *hepEmProcess = new G4HepEmProcess();

  // Add standard EM Processes
  //
  auto aParticleIterator = GetParticleIterator();
  aParticleIterator->reset();
  while ((*aParticleIterator)()) {
    G4ParticleDefinition *particle = aParticleIterator->value();
    G4String particleName          = particle->GetParticleName();

    if (particleName == "gamma") {

      // Add G4HepEm process to gamma: includes Conversion, Compton and photoelectric effect.
      particle->GetProcessManager()->AddProcess(hepEmProcess, -1, -1, 1);

    } else if (particleName == "e-") {

      // Add G4HepEm process to e-: includes Ionisation and Bremsstrahlung for e-
      particle->GetProcessManager()->AddProcess(hepEmProcess, -1, -1, 1);

    } else if (particleName == "e+") {

      // Add G4HepEm process to e+: includes Ionisation, Bremsstrahlung and e+e-
      // annihilation into 2 gamma interactions for e+
      particle->GetProcessManager()->AddProcess(hepEmProcess, -1, -1, 1);
    }
  }

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
