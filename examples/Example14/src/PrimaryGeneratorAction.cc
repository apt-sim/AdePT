// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0
#include "PrimaryGeneratorAction.hh"
#include "PrimaryGeneratorMessenger.hh"
#include "DetectorConstruction.hh"

#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#ifdef HEPMC3_FOUND
#include "HepMC3G4AsciiReader.hh"
#endif

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::PrimaryGeneratorAction(DetectorConstruction *det)
    : G4VUserPrimaryGeneratorAction(), fParticleGun(0), fDetector(det), fRndmBeam(0.), fRndmDirection(0.), fGunMessenger(0),
    fUseHepMC(false)
{
  G4int n_particle = 1;
  fParticleGun     = new G4ParticleGun(n_particle);
  SetDefaultKinematic();

  // create a messenger for this class
  fGunMessenger = new PrimaryGeneratorMessenger(this);

  // if HepMC3, create the reader
  #ifdef HEPMC3_FOUND
  fHepmcAscii = new HepMC3G4AsciiReader();
  #endif
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fParticleGun;
  delete fGunMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryGeneratorAction::GeneratePrimaries(G4Event *aEvent)
{
  // this function is called at the begining of event
  //

  if(fUseHepMC && fHepmcAscii)
  {
    fHepmcAscii->GeneratePrimaryVertex(aEvent);
  }
  else
  {
  G4ThreeVector oldDirection = fParticleGun->GetParticleMomentumDirection();
  // randomize direction if requested
  if(fRndmDirection > 0.) {

    // calculate current phi and eta
    double eta_old = atanh(oldDirection.z());
    double phi_old = atan(oldDirection.y()/oldDirection.z());

    // Generate new phi and new eta in a ranges determined by fRndmDirection parameter
    const double phi = phi_old + (2. * M_PI * G4UniformRand()) * fRndmDirection;
    const double eta = eta_old + (- 5. + 10. * G4UniformRand()) * fRndmDirection;

    // new direction
    G4double dirx = cos(phi) / cosh(eta);
    G4double diry = sin(phi) / cosh(eta);
    G4double dirz = tanh(eta);

    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(dirx,diry,dirz));
  }
  // randomize the beam, if requested.
  if (fRndmBeam > 0.) {
    G4ThreeVector oldPosition = fParticleGun->GetParticlePosition();
    G4double rbeam            = 0.5 * fRndmBeam;
    G4double x0               = oldPosition.x();
    G4double y0               = oldPosition.y() + (2 * G4UniformRand() - 1.) * rbeam;
    G4double z0               = oldPosition.z() + (2 * G4UniformRand() - 1.) * rbeam;
    fParticleGun->SetParticlePosition(G4ThreeVector(x0, y0, z0));
    fParticleGun->GeneratePrimaryVertex(aEvent);
    fParticleGun->SetParticlePosition(oldPosition);
  } else {
    fParticleGun->GeneratePrimaryVertex(aEvent);
  }
  fParticleGun->SetParticleMomentumDirection(oldDirection);
  }
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

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void PrimaryGeneratorAction::Print() const
{
  G4cout << "=== Gun shooting " << fParticleGun->GetParticleDefinition()->GetParticleName() << " with energy "
         << fParticleGun->GetParticleEnergy() / GeV << "[GeV] from: " << fParticleGun->GetParticlePosition() / mm
         << " [mm] along direction: " << fParticleGun->GetParticleMomentumDirection() << "\n";
}
