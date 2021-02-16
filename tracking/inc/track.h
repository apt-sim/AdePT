// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACKH
#define TRACKH

#include <CopCore/Ranluxpp.h>
#include <CopCore/PhysicalConstants.h>

#include <VecGeom/base/Vector3D.h>

#include <VecGeom/navigation/NavStateIndex.h>

#include <cfloat> // for FLT_MAX

enum TrackStatus { alive, dead };

struct track {
  RanluxppDouble rng_state;
  unsigned int index{0};
  double energy{10};
  double numIALeft[3];

  vecgeom::Vector3D<double> pos;
  vecgeom::Vector3D<double> dir;
  vecgeom::NavStateIndex current_state;
  vecgeom::NavStateIndex next_state;

  unsigned int mother_index{0};
  unsigned int eventId{0};
  TrackStatus status{alive};

  float interaction_length{FLT_MAX};
  float total_length{0.0}; // length since track start

  int number_of_secondaries{0}; // total since track start -- used as counter for RNG
  float energy_loss{0};         // current step's eloss -- used by primitive version of scoring

  uint16_t num_step{0}; // number of Steps -- for killing after MaX

  char current_process{0}; // -1 for Navigation / -2 for Field Propagation
  char pdg{0};             // Large enough for e/g. Then uint16_t for p/n

  __device__ __host__ double uniform() { return rng_state.Rndm(); }

  __device__ __host__ int charge() const // charge for e- / e+ / gamma only
  {
    int chrg = (pdg == pdgPositron ? 1 : 0) + (pdg == pdgElectron ? -1 : 0);
    return chrg;
  }

  __device__ __host__ double mass() const // Rest mass for e- / e+ / gamma only
  {
    return (pdg == pdgGamma) ? 0.0 : copcore::units::kElectronMassC2;
  }

  __host__ void print(int id = -1, bool verbose = false) const;

  __device__ __host__ void SwapStates()
  {
    auto state          = this->current_state;
    this->current_state = this->next_state;
    this->next_state    = state;
  }

  static constexpr char pdgElectron = 11;
  static constexpr char pdgPositron = -11;
  static constexpr char pdgGamma    = 22;
};

struct NavigationStateBuffer {
  unsigned int currentTouchIndex;
  unsigned int nextTouchIndex;
  int currentLevel;
  int nextLevel;
};

// inline
__global__ void GetNavStateIndices(const track &trk_dev,                // Unified or device
                                   NavigationStateBuffer &navState_unif // Return => Unified memory
)
{
  // Needed because type of NavStateIndex (in track) is vecgeom:L:cxx::NavStateIndex
  // Solution for printing by Andrei Gheata.
  //
  // trk.current_state.IsOnBoundary();
  // std::cout << " current: "; // <<    setw(3) << next_state.IsOnBoundary();

  auto currentLevel = trk_dev.current_state.GetLevel();

  navState_unif.currentTouchIndex = trk_dev.current_state.GetNavIndex(currentLevel);

  navState_unif.currentLevel = currentLevel;
  // auto currentPhysVolume = trk_dev.current_state.ValueAt(currentLevel);

  // std::cout << " next: "; // <<    setw(3) << next_state.IsOnBoundary();

  auto nextLevel               = trk_dev.next_state.GetLevel(); // This crashes consistently - in gdb 2020.12.17-18
  navState_unif.nextTouchIndex = trk_dev.next_state.GetNavIndex(nextLevel);

  navState_unif.nextLevel = nextLevel;

  // auto nextTouchVolume = trk_dev.next_state.ValueAt(nextLevel);
}

#include <cstdio>
#include <iomanip>

// inline
__host__ void track::print(int extId, bool verbose) const
{
  using std::setw;
  int oldPrec = std::cout.precision(5);

  static std::string particleName[3] = {"e-", "g", "e+"};
  std::cout                                                                      // << " Track "
      << setw(4) << extId << " " << setw(2) << particleName[charge() + 1] << " " // Ok for e-/g/e+ only
      << " / id= " << setw(11) << index
      << " "
      //  << " pdg = " << setw(4) << (int) pdg
      << " m= " << setw(11) << mother_index << " "
      << " ev= " << setw(3) << eventId << " "
      << " step# " << setw(4)
      << (int)num_step
      // << " addr= " << this   << " "
      << " Pos: " << setw(11) << pos[0] << " , " << setw(11) << pos[1] << " , " << setw(11) << pos[2]
      << " s= " << setw(8) << interaction_length << " l= " << setw(8) << total_length << " kE= " << setw(10) << energy
      << " Dir: = " << setw(8) << dir[0] << " , " << setw(8) << dir[1] << " , " << setw(8) << dir[2] << " " << setw(11)
      << (status == alive ? "Alive" : "Dead ") << " "
      << " proc= " << setw(1) << (int)current_process << " ";

  if (verbose) {
    std::cout << "<";
    // current_state.Print();
    // unsigned int currentIndex, nextIndex;
    // char     currentLevel, nextLevel;

    static NavigationStateBuffer *pNavStateBuffer = nullptr;
    if (pNavStateBuffer == nullptr) cudaMallocManaged(&pNavStateBuffer, sizeof(NavigationStateBuffer));

    GetNavStateIndices<<<1, 1>>>(*this, *pNavStateBuffer);
    cudaDeviceSynchronize(); // Needed -- wait for result !!

    std::cout << " current: " << pNavStateBuffer->currentTouchIndex << " "
              << " lv = " << (int)pNavStateBuffer->currentLevel << " " << setw(3)
              << (current_state.IsOnBoundary() ? "bnd" : " in") << " ";

    std::cout << " next: " << pNavStateBuffer->nextTouchIndex << " "
              << " lv = " << pNavStateBuffer->nextLevel << " " << setw(3) << (next_state.IsOnBoundary() ? "bnd" : " in")
              << " ";

    // 2020.12.18 -- caused crash Dec 2020 / Jan 2021 ? - check if still the case
    // current_state.printValueSequence(std::cout);
    // next_state.printValueSequence(std::cout);
    std::cout << ">";
  }

  std::cout << std::endl;
  std::cout.precision(oldPrec);
}

#endif
