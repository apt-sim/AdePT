// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//
#ifndef PARTICLEGUN_HH
#define PARTICLEGUN_HH

#include "G4ParticleGun.hh"
#include "G4Event.hh"

class ParticleGun : public G4ParticleGun {
public:
  ParticleGun() : G4ParticleGun() {}
  ~ParticleGun() {}
  void GenerateRandomPrimaryVertex(G4Event *aEvent, G4double aMinPhi, G4double aMaxPhi, G4double aMinTheta,
                                   G4double aMaxTheta, std::vector<G4ParticleDefinition*> *aParticleList, 
                                   std::vector<float> *aParticleWeights, std::vector<float> *aParticleEnergies);
};

#endif /* PARTICLEGUN_HH */