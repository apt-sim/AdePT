// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//
#include "ParticleGun.hh"
#include "G4ParticleGun.hh"
#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "Randomize.hh"
#include "G4SystemOfUnits.hh"

#include <numeric>

void ParticleGun::GenerateRandomPrimaryVertex(G4Event *aEvent, G4double aMinPhi, G4double aMaxPhi, G4double aMinTheta,
                                              G4double aMaxTheta, std::vector<G4ParticleDefinition *> *aParticleList,
                                              std::vector<float> *aParticleWeights,
                                              std::vector<float> *aParticleEnergies)
{
  // Choose a particle from the list
  float choice = G4UniformRand();
  float weight = 0;

  for (int i = 0; i < aParticleList->size(); i++) {
    weight += (*aParticleWeights)[i];
    if (weight > choice) {
      SetParticleDefinition((*aParticleList)[i]);
      SetParticleEnergy((*aParticleEnergies)[i]);
      break;
    }
  }

  // Create a new vertex
  //
  auto *vertex = new G4PrimaryVertex(particle_position, particle_time);

  // Create new primaries and set them to the vertex
  //
  G4double mass = particle_definition->GetPDGMass();
  for (G4int i = 0; i < NumberOfParticlesToBeGenerated; ++i) {
    auto *particle = new G4PrimaryParticle(particle_definition);

    // Choose a random direction in the selected ranges, with an isotropic distribution
    G4double phi                = (aMaxPhi - aMinPhi) * G4UniformRand() + aMinPhi;
    G4double theta              = acos((cos(aMaxTheta) - cos(aMinTheta)) * G4UniformRand() + cos(aMinTheta));
    G4double x                  = cos(phi) * sin(theta);
    G4double y                  = sin(phi) * sin(theta);
    G4double z                  = cos(theta);
    particle_momentum_direction = G4ThreeVector(x, y, z);

    particle->SetKineticEnergy(particle_energy);
    particle->SetMass(mass);
    particle->SetMomentumDirection(particle_momentum_direction);
    particle->SetCharge(particle_charge);
    particle->SetPolarization(particle_polarization.x(), particle_polarization.y(), particle_polarization.z());
    vertex->SetPrimary(particle);

    // Choose a new particle from the list for the next iteration
    choice = G4UniformRand();
    weight = 0;
    for (int i = 0; i < aParticleList->size(); i++) {
      weight += (*aParticleWeights)[i];
      if (weight > choice) {
        SetParticleDefinition((*aParticleList)[i]);
        SetParticleEnergy((*aParticleEnergies)[i]);
        break;
      }
    }
  }
  aEvent->AddPrimaryVertex(vertex);
}