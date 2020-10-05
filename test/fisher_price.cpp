// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <deque>
#include <iostream>
#include <random>

struct particle

{
  float energy;
};

int main()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rng;

  std::deque<particle> particleStack;

  // Just one particle
  particleStack.emplace_back(particle{100.0f});

  // "Scoring"
  size_t numberOfSecondaries = 0;
  float totalEnergyLoss      = 0;

  while (!particleStack.empty()) {
    auto &p = particleStack.front();
    while (p.energy > 0.0f) {
      // Essentially "process selection"
      float r = rng(gen);

      if (r < 0.5f) {
        // do energy loss
        float eloss = 0.2f * p.energy;
        totalEnergyLoss += (eloss < 0.001f ? p.energy : eloss);
        p.energy = (eloss < 0.001f ? 0.0f : (p.energy - eloss));
      } else {
        // do "pair production"
        numberOfSecondaries++;
        float eloss = 0.5f * p.energy;
        particleStack.emplace_back(particle{eloss});
        p.energy -= eloss;
      }
    }
    // "Kill" particle
    particleStack.pop_front();
  }

  // "Persist"
  std::cout << "Number of secondaries: " << numberOfSecondaries << "\n"
            << "Total energy loss    : " << totalEnergyLoss << "\n";
}
