// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//
#include "ParticleGun.hh"
#include "ParticleGunMessenger.hh"

#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "Randomize.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"

#ifdef HEPMC3_FOUND
#include "HepMC3G4AsciiReader.hh"
#endif

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>

ParticleGun::ParticleGun()
{
  fG4ParticleGun = std::make_unique<G4ParticleGun>();
  // if HepMC3, create the reader
#ifdef HEPMC3_FOUND
  fHepmcAscii = std::make_unique<HepMC3G4AsciiReader>();
#endif

  fMessenger = std::make_unique<ParticleGunMessenger>(this);
}

ParticleGun::~ParticleGun() {}

void ParticleGun::GeneratePrimaries(G4Event *aEvent)
{
  // this function is called at the begining of event
  //

  if (!fUseHepMC && fRandomizeGun && !fInitializationDone) {
    // We only need to do this for the first run
    fInitializationDone = true;
    // Re-balance the user-provided weights if needed
    ReWeight();
    // Make sure all particles have a user-defined energy
    for (unsigned int i = 0; i < fParticleEnergies.size(); i++) {
      if (fParticleEnergies[i] < 0) {
        G4Exception("PrimaryGeneratorAction::GeneratePrimaries()", "Notification", FatalErrorInArgument,
                    ("Energy undefined for  " + fParticleList[i]->GetParticleName()).c_str());
      }
    }
    // In case the upper range for Phi or Theta was not defined, or is lower than the
    // lower range
    if (fMaxPhi < fMinPhi) {
      fMaxPhi = fMinPhi;
    }
    if (fMaxTheta < fMinTheta) {
      fMaxTheta = fMinTheta;
    }
  }

  if (fUseHepMC) {
    if (!fHepmcAscii) throw std::logic_error("ParticleGun: HepMC3 reader is not available.");
    fHepmcAscii->GeneratePrimaryVertex(aEvent);
  } else if (fRandomizeGun) {
    GenerateRandomPrimaryVertex(aEvent, fMinPhi, fMaxPhi, fMinTheta, fMaxTheta, &fParticleList, &fParticleWeights,
                                &fParticleEnergies);
  } else {
    fG4ParticleGun->GeneratePrimaryVertex(aEvent);
  }

  // Print the particle gun info if requested
  if (fPrintGun) Print();

  PrintPrimaries(aEvent);
}

void ParticleGun::GenerateRandomPrimaryVertex(G4Event *aEvent, G4double aMinPhi, G4double aMaxPhi, G4double aMinTheta,
                                              G4double aMaxTheta, std::vector<G4ParticleDefinition *> *aParticleList,
                                              std::vector<float> *aParticleWeights,
                                              std::vector<float> *aParticleEnergies)
{
  // Choose a particle from the list
  float choice = G4UniformRand();
  float weight = 0;

  for (unsigned int i = 0; i < aParticleList->size(); i++) {
    weight += (*aParticleWeights)[i];
    if (weight > choice) {
      fG4ParticleGun->SetParticleDefinition((*aParticleList)[i]);
      fG4ParticleGun->SetParticleEnergy((*aParticleEnergies)[i]);
      break;
    }
  }

  // Create a new vertex
  //
  auto *vertex = new G4PrimaryVertex(fG4ParticleGun->GetParticlePosition(), fG4ParticleGun->GetParticleTime());

  // Create new primaries and set them to the vertex
  //
  G4double mass = fG4ParticleGun->GetParticleDefinition()->GetPDGMass();
  for (G4int i = 0; i < fG4ParticleGun->GetNumberOfParticles(); ++i) {
    auto *particle = new G4PrimaryParticle(fG4ParticleGun->GetParticleDefinition());

    // Choose a random direction in the selected ranges, with an isotropic distribution
    G4double phi   = (aMaxPhi - aMinPhi) * G4UniformRand() + aMinPhi;
    G4double theta = acos((cos(aMaxTheta) - cos(aMinTheta)) * G4UniformRand() + cos(aMinTheta));
    G4double x     = cos(phi) * sin(theta);
    G4double y     = sin(phi) * sin(theta);
    G4double z     = cos(theta);
    fG4ParticleGun->SetParticleMomentumDirection(G4ThreeVector(x, y, z));

    particle->SetKineticEnergy(fG4ParticleGun->GetParticleEnergy());
    particle->SetMass(mass);
    particle->SetMomentumDirection(fG4ParticleGun->GetParticleMomentumDirection());
    particle->SetCharge(fG4ParticleGun->GetParticleCharge());
    particle->SetPolarization(fG4ParticleGun->GetParticlePolarization());
    vertex->SetPrimary(particle);

    // Choose a new particle from the list for the next iteration
    choice = G4UniformRand();
    weight = 0;
    for (unsigned int i = 0; i < aParticleList->size(); i++) {
      weight += (*aParticleWeights)[i];
      if (weight > choice) {
        fG4ParticleGun->SetParticleDefinition((*aParticleList)[i]);
        fG4ParticleGun->SetParticleEnergy((*aParticleEnergies)[i]);
        break;
      }
    }
  }
  aEvent->AddPrimaryVertex(vertex);
}

void ParticleGun::SetDefaultKinematic()
{
  G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  G4ParticleDefinition *particle = particleTable->FindParticle(particleName = "e-");
  fG4ParticleGun->SetParticleDefinition(particle);
  fG4ParticleGun->SetParticleMomentumDirection(G4ThreeVector(1., 1., 1.));
  fG4ParticleGun->SetParticleEnergy(1. * GeV);
  G4double position = 0.0;
  fG4ParticleGun->SetParticlePosition(G4ThreeVector(position, 0. * cm, 0. * cm));
}

void ParticleGun::AddParticle(G4ParticleDefinition *val, float weight, double energy)
{
  fParticleList.push_back(val);
  fParticleWeights.push_back(weight);
  fParticleEnergies.push_back(energy);
}

void ParticleGun::ReWeight()
{
  double userDefinedSum = 0;
  double numNotDefined  = 0;
  for (float i : fParticleWeights)
    i >= 0 ? userDefinedSum += i : numNotDefined += 1;

  if (userDefinedSum < 1 && numNotDefined == 0) {
    // If the user-provided weights do not sum up to 1 and there are no particles left to
    // distribute the remaining weight, re-balance their weights
    for (unsigned int i = 0; i < fParticleWeights.size(); i++) {
      fParticleWeights[i] = fParticleWeights[i] / userDefinedSum;
      G4Exception("PrimaryGeneratorAction::ReWeight()", "Notification", JustWarning,
                  ("Sum of user-defined weights is <1, new weight for " + fParticleList[i]->GetParticleName() + " = " +
                   std::to_string(fParticleWeights[i]))
                      .c_str());
    }
  } else {
    for (unsigned int i = 0; i < fParticleWeights.size(); i++) {
      double originalWeight = fParticleWeights[i];
      // Particles with no user-defined weight have weight -1
      if (originalWeight >= 0) {
        // For particles with user-defined weight, re-balance only if the sum is higher than 1
        if (userDefinedSum <= 1) {
          fParticleWeights[i] = originalWeight;
        } else {
          fParticleWeights[i] = originalWeight / userDefinedSum;
          G4Exception("PrimaryGeneratorAction::ReWeight()", "Notification", JustWarning,
                      ("Sum of user-defined weights is >1, new weight for " + fParticleList[i]->GetParticleName() +
                       " = " + std::to_string(fParticleWeights[i]))
                          .c_str());
        }
      } else if (userDefinedSum < 1) {
        // For particles with no user-defined weight, distribute the remaining weight
        fParticleWeights[i] = (1 - userDefinedSum) / numNotDefined;
      } else {
        // If the sum of user-defined weights is greater or equal to 1 there's nothing left to distribute,
        // the probability for the remaining particles will be 0
        fParticleWeights[i] = 0;
      }
    }
  }
}

void ParticleGun::Print()
{
  if (!fRandomizeGun) {
    G4cout << "=== Gun shooting " << fG4ParticleGun->GetParticleDefinition()->GetParticleName() << " with energy "
           << fG4ParticleGun->GetParticleEnergy() / GeV << "[GeV] from: " << fG4ParticleGun->GetParticlePosition() / mm
           << " [mm] along direction: " << fG4ParticleGun->GetParticleMomentumDirection() << G4endl;
  } else {
    for (unsigned int i = 0; i < fParticleList.size(); i++) {
      G4cout << "=== Gun shooting " << fParticleEnergies[i] / GeV << "[GeV] " << fParticleList[i]->GetParticleName()
             << " with probability " << fParticleWeights[i] * 100 << "%" << G4endl;
    }
    G4cout << "=== Gun shooting from: " << fG4ParticleGun->GetParticlePosition() / mm << " [mm]" << G4endl;
    G4cout << "=== Gun shooting in ranges: " << G4endl;
    G4cout << "Phi: [" << fMinPhi << ", " << fMaxPhi << "] (rad)" << G4endl;
    G4cout << "Theta: [" << fMinTheta << ", " << fMaxTheta << "] (rad)" << G4endl;
  }
}

void ParticleGun::PrintPrimaries(G4Event *aEvent) const
{
  struct ParticleData {
    std::string name;
    unsigned int count{0};
    double energy{0.};
    bool operator<(const ParticleData &other) const
    {
      if (energy != other.energy) return energy > other.energy;
      if (count != other.count) return count > other.count;
      return name < other.name;
    }
  };
  std::map<G4String, ParticleData> aParticleCounts;

  const auto nVtx = aEvent->GetNumberOfPrimaryVertex();
  for (int vtx = 0; vtx < nVtx; ++vtx) {
    const G4PrimaryVertex *vertex = aEvent->GetPrimaryVertex(vtx);
    const auto nParticle          = vertex->GetNumberOfParticle();
    for (int i = 0; i < nParticle; i++) {
      const G4PrimaryParticle *particle = vertex->GetPrimary(i);
      G4String aParticleName            = particle->GetParticleDefinition()->GetParticleName();
      G4double aParticleEnergy          = particle->GetKineticEnergy();
      auto &particleTuple               = aParticleCounts[aParticleName];
      particleTuple.name                = aParticleName;
      particleTuple.count++;
      particleTuple.energy += aParticleEnergy;
    }
  }

  std::vector<ParticleData> vec(aParticleCounts.size());
  std::transform(aParticleCounts.begin(), aParticleCounts.end(), vec.begin(), [](auto &pair) { return pair.second; });
  std::sort(vec.begin(), vec.end());

  std::stringstream message;
  message << "Event: " << aEvent->GetEventID() << " NVertex: " << nVtx << "\n";
  for (auto const &pd : vec) {
    message << std::setw(20) << pd.name << ": " << std::setw(9) << pd.count << " eKin (GeV): " << std::setw(15)
            << pd.energy / GeV << " (total) " << std::setw(15) << pd.energy / pd.count / GeV << " (avg)\n";
  }
  message << "Total energy: " << std::setw(9)
          << std::accumulate(vec.begin(), vec.end(), 0.,
                             [](double sum, auto const &data) { return sum + data.energy; }) /
                 GeV
          << " GeV\n";
  G4cout << message.str() << G4endl;
}
