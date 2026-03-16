// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRUTHHISTOGRAMMER_HH
#define TRUTHHISTOGRAMMER_HH

#include <cstdint>
#include <map>
#include <string>

class G4Step;
class G4Track;

/**
 * @brief Collects aggregated truth observables for the ROOT-based drift test.
 *
 * The collector stores categorical histograms as label -> count maps and
 * continuous observables as exact floating-point bit patterns -> count maps.
 * The master thread later serializes these maps into ROOT histograms in a
 * deterministic order for exact comparison against the reference file.
 *
 * Categorical labels are written into sidecar ROOT metadata objects instead of
 * TH1 axis labels. This keeps the semantic comparison exact while avoiding the
 * expensive ROOT label bookkeeping in the multi-threaded runs.
 */
class TruthHistogrammer {
public:
  /**
   * @brief Exact weighted energy-deposit contributions accumulated for one physical volume.
   *
   * The ROOT truth path does not sum deposits on the fly. Instead it stores the
   * exact bit pattern of each deposited `double` value and counts how often that
   * value occurred. Worker-thread merging is then an integer-count merge over
   * identical bit patterns, which is order-independent.
   */
  struct VolumeEdep {
    std::string label;
    std::map<std::uint64_t, std::uint64_t> contributions;
  };

  /// Histogram with stable string labels and integer counts.
  using CountHistogram = std::map<std::string, std::uint64_t>;
  /**
   * @brief Histogram with exact `double` bit patterns and integer counts.
   *
   * This is the key trick that makes the ROOT truth path reproducible in MT:
   * floating-point values are first bucketed by their exact IEEE-754 bit
   * pattern, and only the integer populations are merged across workers.
   */
  using ValueHistogram = std::map<std::uint64_t, std::uint64_t>;

  TruthHistogrammer();

  /// Record the initial state of a track after its lineage information is available.
  void RecordInitialTrack(const G4Track *track);
  /// Record the final state of a track using the last reconstructed step.
  void RecordFinalTrack(const G4Track *track);
  /// Record per-step observables for every reconstructed step.
  void RecordStep(const G4Step *step);
  /// Record the population of the resolved primary ancestor IDs.
  void RecordPrimaryAncestorPopulation(int primaryTrackID);
  /// Record the population of the resolved shower generations.
  void RecordGenerationPopulation(unsigned int generation);
  /// Record deposited energy per physical volume, matching the SD-based CSV path.
  void AddEnergyDeposit(int physicalVolumeId, const std::string &physicalVolumeName, double energyDeposit);
  /// Merge another worker-thread collector into this collector.
  void MergeFrom(const TruthHistogrammer &other);
  /// Serialize all collected observables into a deterministic ROOT file.
  void WriteROOTFile(const std::string &path) const;

private:
  std::map<std::string, CountHistogram> fCategoricalHistograms;
  std::map<std::string, ValueHistogram> fValueHistograms;
  std::map<int, VolumeEdep> fEnergyDepositByVolume;
  void IncrementCategorical(const std::string &histogramName, const std::string &label, std::uint64_t count = 1);
  /// Store one exact floating-point observation by its raw bit pattern.
  void IncrementValue(const std::string &histogramName, double value, std::uint64_t count = 1);
};

#endif
