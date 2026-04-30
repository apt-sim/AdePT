// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_TRANSPORT_CONFIG_HH
#define ADEPT_TRANSPORT_CONFIG_HH

#include <cstdint>
#include <string>

/// @brief Plain configuration values required to initialize the AdePT transport engine.
struct AdePTTransportConfig {
  uint64_t adeptSeed{1234567};
  unsigned short numThreads{0};
  unsigned int trackCapacity{0};
  unsigned int scoringCapacity{0};
  int debugLevel{0};
  int cudaStackLimit{0};
  int cudaHeapLimit{0};
  unsigned short lastNParticlesOnCPU{0};
  unsigned short maxWDTIter{5};
  bool returnAllSteps{false};
  bool returnFirstAndLastStep{false};
  std::string bfieldFile{};
  double cpuCapacityFactor{2.5};
  double cpuCopyFraction{0.5};
  double hitBufferSafetyFactor{1.5};
};

#endif
