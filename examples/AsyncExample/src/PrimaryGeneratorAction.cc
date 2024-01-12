// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
#include "PrimaryGeneratorAction.hh"
#include "PrimaryGeneratorMessenger.hh"
#include "DetectorConstruction.hh"

#include "ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#ifdef HEPMC3_FOUND
#include "HepMC3G4AsciiReader.hh"
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::PrimaryGeneratorAction(DetectorConstruction *det)
    : G4VUserPrimaryGeneratorAction(), fParticleGun(0), fDetector(det), fRndmBeam(0.), fRndmDirection(0.),
      fGunMessenger(0), fUseHepMC(false), fRandomizeGun(false), fInitializationDone(false), fPrintGun(false),
      fMinPhi(0), fMaxPhi(0), fMinTheta(0), fMaxTheta(0)
{
  G4int n_particle = 1;
  fParticleGun     = new ParticleGun();
  SetDefaultKinematic();

  // create a messenger for this class
  fGunMessenger = new PrimaryGeneratorMessenger(this);

// if HepMC3, create the reader
#ifdef HEPMC3_FOUND
  fHepmcAscii = new HepMC3G4AsciiReader();
#endif

  fParticleList     = new std::vector<G4ParticleDefinition *>();
  fParticleWeights  = new std::vector<float>();
  fParticleEnergies = new std::vector<float>();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
  delete fGunMessenger;
  delete fParticleList;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryGeneratorAction::GeneratePrimaries(G4Event *aEvent)
{
  // this function is called at the begining of event
  //

  if (fRandomizeGun) {
    if (!fInitializationDone) {
      // We only need to do this for the first run
      fInitializationDone = true;
      // Re-balance the user-provided weights if needed
      ReWeight();
      // Make sure all particles have a user-defined energy
      for(int i=0; i<fParticleEnergies->size(); i++)
      {
        if((*fParticleEnergies)[i] < 0)
        {
          G4Exception("PrimaryGeneratorAction::GeneratePrimaries()", "Notification", FatalErrorInArgument,
                  ("Energy undefined for  " + (*fParticleList)[i]->GetParticleName()).c_str());
        }
      }
      // In case the upper range for Phi or Theta was not defined, or is lower than the 
      // lower range
      if(fMaxPhi < fMinPhi)
      {
        fMaxPhi = fMinPhi;
      }
      if(fMaxTheta < fMinTheta)
      {
        fMaxTheta = fMinTheta;
      }
    }
  }

  if (fUseHepMC && fHepmcAscii) {
    fHepmcAscii->GeneratePrimaryVertex(aEvent);
  } else {
    G4ThreeVector oldDirection = fParticleGun->GetParticleMomentumDirection();
    // randomize direction if requested
    if (fRndmDirection > 0.) {

      // calculate current phi and eta
      double eta_old = atanh(oldDirection.z());
      double phi_old = atan(oldDirection.y() / oldDirection.z());

      // Generate new phi and new eta in a ranges determined by fRndmDirection parameter
      const double phi = phi_old + (2. * M_PI * G4UniformRand()) * fRndmDirection;
      const double eta = eta_old + (-5. + 10. * G4UniformRand()) * fRndmDirection;

      // new direction
      G4double dirx = cos(phi) / cosh(eta);
      G4double diry = sin(phi) / cosh(eta);
      G4double dirz = tanh(eta);

      fParticleGun->SetParticleMomentumDirection(G4ThreeVector(dirx, diry, dirz));
    }
    // randomize the beam, if requested.
    if (fRndmBeam > 0.) {
      G4ThreeVector oldPosition = fParticleGun->GetParticlePosition();
      G4double rbeam            = 0.5 * fRndmBeam;
      G4double x0               = oldPosition.x();
      G4double y0               = oldPosition.y() + (2 * G4UniformRand() - 1.) * rbeam;
      G4double z0               = oldPosition.z() + (2 * G4UniformRand() - 1.) * rbeam;
      fParticleGun->SetParticlePosition(G4ThreeVector(x0, y0, z0));
      if (fRandomizeGun) {
        fParticleGun->GenerateRandomPrimaryVertex(aEvent, fMinPhi, fMaxPhi, fMinTheta, fMaxTheta, fParticleList,
                                                  fParticleWeights, fParticleEnergies);
      } else {
        fParticleGun->GeneratePrimaryVertex(aEvent);
      }
      fParticleGun->SetParticlePosition(oldPosition);
    } else {
      if (fRandomizeGun) {
        fParticleGun->GenerateRandomPrimaryVertex(aEvent, fMinPhi, fMaxPhi, fMinTheta, fMaxTheta, fParticleList,
                                                  fParticleWeights, fParticleEnergies);
      } else {
        fParticleGun->GeneratePrimaryVertex(aEvent);
      }
    }
    fParticleGun->SetParticleMomentumDirection(oldDirection);
  }

  // Print the particle gun info if requested
  if (fPrintGun) Print();

  PrintPrimaries(aEvent);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryGeneratorAction::SetDefaultKinematic()
{
  G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  G4ParticleDefinition *particle = particleTable->FindParticle(particleName = "e-");
  fParticleGun->SetParticleDefinition(particle);
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(1., 1., 1.));
  fParticleGun->SetParticleEnergy(1. * GeV);
  G4double position = 0.0;
  fParticleGun->SetParticlePosition(G4ThreeVector(position, 0. * cm, 0. * cm));
}

void PrimaryGeneratorAction::AddParticle(G4ParticleDefinition *val, float weight, double energy)
{
  fParticleList->push_back(val);
  fParticleWeights->push_back(weight);
  fParticleEnergies->push_back(energy);
}

void PrimaryGeneratorAction::ReWeight()
{
  double userDefinedSum = 0;
  double numNotDefined  = 0;
  for (float i : *fParticleWeights)
    i >= 0 ? userDefinedSum += i : numNotDefined += 1;

  if (userDefinedSum < 1 && numNotDefined == 0) {
    // If the user-provided weights do not sum up to 1 and there are no particles left to
    // distribute the remaining weight, re-balance their weights
    for (int i = 0; i < fParticleWeights->size(); i++) {
      (*fParticleWeights)[i] = (*fParticleWeights)[i] / userDefinedSum;
      G4Exception("PrimaryGeneratorAction::ReWeight()", "Notification", JustWarning,
                  ("Sum of user-defined weights is <1, new weight for " + (*fParticleList)[i]->GetParticleName() +
                   " = " + std::to_string((*fParticleWeights)[i]))
                      .c_str());
    }
  } else {
    for (int i = 0; i < fParticleWeights->size(); i++) {
      double originalWeight = (*fParticleWeights)[i];
      // Particles with no user-defined weight have weight -1
      if (originalWeight >= 0) {
        // For particles with user-defined weight, re-balance only if the sum is higher than 1
        if (userDefinedSum <= 1) {
          (*fParticleWeights)[i] = originalWeight;
        } else {
          (*fParticleWeights)[i] = originalWeight / userDefinedSum;
          G4Exception("PrimaryGeneratorAction::ReWeight()", "Notification", JustWarning,
                      ("Sum of user-defined weights is >1, new weight for " + (*fParticleList)[i]->GetParticleName() +
                       " = " + std::to_string((*fParticleWeights)[i]))
                          .c_str());
        }
      } else if (userDefinedSum < 1) {
        // For particles with no user-defined weight, distribute the remaining weight
        (*fParticleWeights)[i] = (1 - userDefinedSum) / numNotDefined;
      } else {
        // If the sum of user-defined weights is greater or equal to 1 there's nothing left to distribute,
        // the probability for the remaining particles will be 0
        (*fParticleWeights)[i] = 0;
      }
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PrimaryGeneratorAction::Print() const
{
  if (!fRandomizeGun) {
    G4cout << "=== Gun shooting " << fParticleGun->GetParticleDefinition()->GetParticleName() << " with energy "
           << fParticleGun->GetParticleEnergy() / GeV << "[GeV] from: " << fParticleGun->GetParticlePosition() / mm
           << " [mm] along direction: " << fParticleGun->GetParticleMomentumDirection() << G4endl;
  } else {
    for (int i = 0; i < fParticleList->size(); i++) {
      G4cout << "=== Gun shooting " << (*fParticleEnergies)[i] / GeV << "[GeV] "
             << (*fParticleList)[i]->GetParticleName() << " with probability " << (*fParticleWeights)[i] * 100 << "%"
             << G4endl;
    }
    G4cout << "=== Gun shooting from: " << fParticleGun->GetParticlePosition() / mm << " [mm]" << G4endl;
    G4cout << "=== Gun shooting in ranges: " << G4endl;
    G4cout << "Phi: [" << fMinPhi << ", " << fMaxPhi << "] (rad)" << G4endl;
    G4cout << "Theta: [" << fMinTheta << ", " << fMaxTheta << "] (rad)" << G4endl;
  }
}

void PrimaryGeneratorAction::PrintPrimaries(G4Event* aEvent) const
{
  std::map<G4String, G4int> aParticleCounts = {};
  std::map<G4String, G4double> aParticleAverageEnergies = {};

  for(int i=0; i<aEvent->GetPrimaryVertex()->GetNumberOfParticle(); i++)
  {
    G4String aParticleName = aEvent->GetPrimaryVertex()->GetPrimary(i)->GetParticleDefinition()->GetParticleName();
    G4double aParticleEnergy = aEvent->GetPrimaryVertex()->GetPrimary(i)->GetKineticEnergy();
    if(!aParticleCounts.count(aParticleName))
    {
      aParticleCounts[aParticleName] = 0;
      aParticleAverageEnergies[aParticleName] = 0;
    }
    aParticleCounts[aParticleName] += 1;
    aParticleAverageEnergies[aParticleName] += aParticleEnergy;
  }
    
  for(auto pd : aParticleCounts)
  {
    G4cout << pd.first << ": " << pd.second << ", " << aParticleAverageEnergies[pd.first]/pd.second << G4endl;
  }
}
