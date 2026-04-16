// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include "TruthHistogrammer.hh"

#include "G4LogicalVolume.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VProcess.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef ADEPT_INTEGRATIONTEST_HAS_ROOT
#include "TFile.h"
#include "TH1D.h"
#include "TObjString.h"
#endif

namespace {

constexpr const char *kEmptyBinLabel = "__empty__";

using CountHistogram   = TruthHistogrammer::CountHistogram;
using IntegerHistogram = TruthHistogrammer::IntegerHistogram;
using ValueHistogram   = TruthHistogrammer::ValueHistogram;

const std::vector<std::string> &CategoricalHistogramNames()
{
  static const std::vector<std::string> names = {
      "creator_process_counts", "step_defining_process_counts", "step_particle_type_counts",
      "step_volume_counts",     "particle_type_counts",         "initial_volume_counts",
      "final_volume_counts",    "vertex_logical_volume_counts",
  };
  return names;
}

const std::vector<std::string> &IntegerHistogramNames()
{
  static const std::vector<std::string> names = {"primary_ancestor_population", "generation_population"};
  return names;
}

const std::vector<std::string> &ValueHistogramNames()
{
  static const std::vector<std::string> names = {
      "initial_x",
      "initial_y",
      "initial_z",
      "initial_dir_x",
      "initial_dir_y",
      "initial_dir_z",
      "initial_ekin",
      "initial_global_time",
      "step_length",
      "step_total_edep",
      "step_pre_ekin",
      "step_global_time",
      "final_x",
      "final_y",
      "final_z",
      "final_dir_x",
      "final_dir_y",
      "final_dir_z",
      "final_ekin",
      "final_global_time",
      "final_local_time",
      "final_proper_time",
      "final_step_length",
      "final_total_edep",
      "final_num_secondaries",
      "vertex_x",
      "vertex_y",
      "vertex_z",
      "vertex_dir_x",
      "vertex_dir_y",
      "vertex_dir_z",
      "vertex_ekin",
  };
  return names;
}

std::string VolumeName(const G4LogicalVolume *volume)
{
  return volume ? volume->GetName() : "none";
}

std::string PhysicalVolumeLabel(const G4Track *track, const std::string &fallback)
{
  if (track == nullptr) return fallback;

  const auto *volume = track->GetVolume();
  if (volume == nullptr) return fallback;

  std::ostringstream stream;
  stream << volume->GetInstanceID() << ":" << volume->GetName();
  return stream.str();
}

std::string PhysicalVolumeLabel(const G4StepPoint *stepPoint, const std::string &fallback)
{
  if (stepPoint == nullptr) return fallback;

  const auto *volume = stepPoint->GetPhysicalVolume();
  if (volume == nullptr) return fallback;

  std::ostringstream stream;
  stream << volume->GetInstanceID() << ":" << volume->GetName();
  return stream.str();
}

std::string ProcessName(const G4VProcess *process)
{
  return process ? process->GetProcessName() : "none";
}

std::string ParticleName(const G4Track *track)
{
  return track && track->GetParticleDefinition() ? track->GetParticleDefinition()->GetParticleName() : "none";
}

#ifdef ADEPT_INTEGRATIONTEST_HAS_ROOT
std::string ExactValueLabel(double value)
{
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value > 0.0 ? "inf" : "-inf";

  std::ostringstream stream;
  stream << std::hexfloat << value;
  return stream.str();
}

bool ValueKeyLess(double lhs, double rhs)
{
  if (std::isnan(lhs)) return !std::isnan(rhs);
  if (std::isnan(rhs)) return false;
  if (lhs < rhs) return true;
  if (rhs < lhs) return false;
  return false;
}

void WriteLabelMetadata(TFile &file, const std::string &name, const std::vector<std::string> &labels)
{
  std::ostringstream metadata;
  if (labels.empty()) {
    metadata << kEmptyBinLabel << "\n";
  } else {
    for (const auto &label : labels) {
      metadata << label << "\n";
    }
  }

  TObjString metadataObject(metadata.str().c_str());
  metadataObject.Write((name + "__labels").c_str());
}

void WriteCountHistogram(TFile &file, const std::string &name, const CountHistogram &histogram)
{
  const int binCount = histogram.empty() ? 1 : static_cast<int>(histogram.size());
  TH1D hist(name.c_str(), name.c_str(), binCount, 0.5, binCount + 0.5);
  hist.SetDirectory(&file);

  if (histogram.empty()) {
    hist.SetBinContent(1, 0.0);
    WriteLabelMetadata(file, name, {});
  } else {
    std::vector<std::string> labels;
    labels.reserve(histogram.size());
    int bin = 1;
    for (const auto &[label, count] : histogram) {
      hist.SetBinContent(bin, static_cast<double>(count));
      labels.push_back(label);
      ++bin;
    }
    WriteLabelMetadata(file, name, labels);
  }

  hist.Write();
}

void WriteIntegerHistogram(TFile &file, const std::string &name, const IntegerHistogram &histogram)
{
  if (histogram.empty()) {
    TH1D hist(name.c_str(), name.c_str(), 1, -0.5, 0.5);
    hist.SetDirectory(&file);
    hist.SetBinContent(1, 0.0);
    hist.Write();
    return;
  }

  const int minValue = histogram.begin()->first;
  const int maxValue = histogram.rbegin()->first;
  const int binCount = maxValue - minValue + 1;
  TH1D hist(name.c_str(), name.c_str(), binCount, minValue - 0.5, maxValue + 0.5);
  hist.SetDirectory(&file);

  for (const auto &[value, count] : histogram) {
    hist.SetBinContent(hist.FindBin(static_cast<double>(value)), static_cast<double>(count));
  }

  hist.Write();
}

void WriteValueHistogram(TFile &file, const std::string &name, const ValueHistogram &histogram)
{
  // The TH1 stores only the population of each distinct exact value. The exact
  // values themselves are written into sidecar metadata as hex-float strings.
  //
  // This avoids two problems:
  // 1. default decimal formatting can be lossy or platform-sensitive
  // 2. ROOT bin labels were too expensive / fragile for the MT truth output
  const int binCount = histogram.empty() ? 1 : static_cast<int>(histogram.size());
  TH1D hist(name.c_str(), name.c_str(), binCount, 0.5, binCount + 0.5);
  hist.SetDirectory(&file);

  if (histogram.empty()) {
    hist.SetBinContent(1, 0.0);
    TObjString metadata(kEmptyBinLabel);
    metadata.Write((name + "__values").c_str());
  } else {
    std::vector<double> keys;
    keys.reserve(histogram.size());
    for (const auto &[value, _] : histogram) {
      keys.push_back(value);
    }
    std::sort(keys.begin(), keys.end(), ValueKeyLess);

    std::ostringstream metadata;
    int bin = 1;
    for (double value : keys) {
      hist.SetBinContent(bin, static_cast<double>(histogram.at(value)));
      metadata << ExactValueLabel(value) << "\n";
      ++bin;
    }

    TObjString metadataObject(metadata.str().c_str());
    metadataObject.Write((name + "__values").c_str());
  }

  hist.Write();
}

void WriteEnergyDepositHistogram(TFile &file, const std::map<int, TruthHistogrammer::VolumeEdep> &energyDepositByVolume)
{
  const int binCount = energyDepositByVolume.empty() ? 1 : static_cast<int>(energyDepositByVolume.size());
  TH1D hist("edep_by_volume", "edep_by_volume", binCount, 0.5, binCount + 0.5);
  hist.SetDirectory(&file);

  if (energyDepositByVolume.empty()) {
    hist.SetBinContent(1, 0.0);
    WriteLabelMetadata(file, "edep_by_volume", {});
  } else {
    std::vector<std::string> labels;
    labels.reserve(energyDepositByVolume.size());
    int bin = 1;
    for (const auto &[volumeId, entry] : energyDepositByVolume) {
      std::ostringstream label;
      label << volumeId << ":" << entry.label;
      hist.SetBinContent(bin, entry.total);
      labels.push_back(label.str());
      ++bin;
    }
    WriteLabelMetadata(file, "edep_by_volume", labels);
  }

  hist.Write();
}
#endif

} // namespace

TruthHistogrammer::TruthHistogrammer()
{
  // Materialize the histogram inventory up front so both the produced ROOT file
  // and the comparison script see the same set of observables, even when a run
  // leaves some of them empty.
  for (const auto &name : CategoricalHistogramNames()) {
    fCategoricalHistograms.emplace(name, CountHistogram{});
  }
  for (const auto &name : IntegerHistogramNames()) {
    fIntegerHistograms.emplace(name, IntegerHistogram{});
  }
  for (const auto &name : ValueHistogramNames()) {
    fValueHistograms.emplace(name, ValueHistogram{});
  }
}

void TruthHistogrammer::IncrementCategorical(const std::string &histogramName, const std::string &label,
                                             std::uint64_t count)
{
  fCategoricalHistograms[histogramName][label] += count;
}

void TruthHistogrammer::IncrementInteger(const std::string &histogramName, int value, std::uint64_t count)
{
  fIntegerHistograms[histogramName][value] += count;
}

void TruthHistogrammer::IncrementValue(const std::string &histogramName, double value, std::uint64_t count)
{
  fValueHistograms[histogramName][value] += count;
}

void TruthHistogrammer::RecordInitialTrack(const G4Track *track)
{
  if (track == nullptr) return;

  IncrementCategorical("creator_process_counts", ProcessName(track->GetCreatorProcess()));
  IncrementCategorical("particle_type_counts", ParticleName(track));
  IncrementCategorical("initial_volume_counts", PhysicalVolumeLabel(track, "none"));
  IncrementCategorical("vertex_logical_volume_counts", VolumeName(track->GetLogicalVolumeAtVertex()));

  IncrementValue("initial_x", track->GetPosition().x());
  IncrementValue("initial_y", track->GetPosition().y());
  IncrementValue("initial_z", track->GetPosition().z());
  IncrementValue("initial_dir_x", track->GetMomentumDirection().x());
  IncrementValue("initial_dir_y", track->GetMomentumDirection().y());
  IncrementValue("initial_dir_z", track->GetMomentumDirection().z());
  IncrementValue("initial_ekin", track->GetKineticEnergy());
  IncrementValue("initial_global_time", track->GetGlobalTime());

  IncrementValue("vertex_x", track->GetVertexPosition().x());
  IncrementValue("vertex_y", track->GetVertexPosition().y());
  IncrementValue("vertex_z", track->GetVertexPosition().z());
  IncrementValue("vertex_dir_x", track->GetVertexMomentumDirection().x());
  IncrementValue("vertex_dir_y", track->GetVertexMomentumDirection().y());
  IncrementValue("vertex_dir_z", track->GetVertexMomentumDirection().z());
  IncrementValue("vertex_ekin", track->GetVertexKineticEnergy());
}

void TruthHistogrammer::RecordFinalTrack(const G4Track *track)
{
  if (track == nullptr) return;

  const auto *step = track->GetStep();
  if (step == nullptr) return;

  const auto *preStep  = step->GetPreStepPoint();
  const auto *postStep = step->GetPostStepPoint();
  if (preStep == nullptr || postStep == nullptr) return;

  // Use the volume traversed by the last step rather than the post-step volume.
  // For boundary-ending steps, the post-step assignment can flip between
  // adjacent volumes while the physical outcome is unchanged.
  IncrementCategorical("final_volume_counts", PhysicalVolumeLabel(preStep, "OUTSIDE_WORLD"));

  IncrementValue("final_x", postStep->GetPosition().x());
  IncrementValue("final_y", postStep->GetPosition().y());
  IncrementValue("final_z", postStep->GetPosition().z());
  IncrementValue("final_dir_x", postStep->GetMomentumDirection().x());
  IncrementValue("final_dir_y", postStep->GetMomentumDirection().y());
  IncrementValue("final_dir_z", postStep->GetMomentumDirection().z());
  IncrementValue("final_ekin", postStep->GetKineticEnergy());
  IncrementValue("final_global_time", postStep->GetGlobalTime());
  IncrementValue("final_local_time", track->GetLocalTime());
  IncrementValue("final_proper_time", track->GetProperTime());
  IncrementValue("final_step_length", step->GetStepLength());
  IncrementValue("final_total_edep", step->GetTotalEnergyDeposit());

  const auto *secondaries = step->GetSecondaryInCurrentStep();
  IncrementValue("final_num_secondaries", secondaries ? static_cast<double>(secondaries->size()) : 0.0);
}

void TruthHistogrammer::RecordStep(const G4Step *step)
{
  if (step == nullptr || step->GetPreStepPoint() == nullptr || step->GetPostStepPoint() == nullptr) return;

  IncrementValue("step_length", step->GetStepLength());
  IncrementValue("step_total_edep", step->GetTotalEnergyDeposit());
  IncrementValue("step_pre_ekin", step->GetPreStepPoint()->GetKineticEnergy());
  IncrementValue("step_global_time", step->GetPreStepPoint()->GetGlobalTime());
  IncrementCategorical("step_defining_process_counts", ProcessName(step->GetPostStepPoint()->GetProcessDefinedStep()));
  IncrementCategorical("step_particle_type_counts", ParticleName(step->GetTrack()));
  IncrementCategorical("step_volume_counts", PhysicalVolumeLabel(step->GetPreStepPoint(), "OUTSIDE_WORLD"));
}

void TruthHistogrammer::RecordPrimaryAncestorPopulation(int primaryTrackID)
{
  IncrementInteger("primary_ancestor_population", primaryTrackID);
}

void TruthHistogrammer::RecordGenerationPopulation(unsigned int generation)
{
  IncrementInteger("generation_population", static_cast<int>(generation));
}

void TruthHistogrammer::AddEnergyDeposit(int physicalVolumeId, const std::string &physicalVolumeName,
                                         double energyDeposit)
{
  auto &entry = fEnergyDepositByVolume[physicalVolumeId];
  entry.label = physicalVolumeName;
  entry.total += energyDeposit;
}

void TruthHistogrammer::MergeFrom(const TruthHistogrammer &other)
{
  for (const auto &[histogramName, histogram] : other.fCategoricalHistograms) {
    auto &target = fCategoricalHistograms[histogramName];
    for (const auto &[label, count] : histogram) {
      target[label] += count;
    }
  }

  for (const auto &[histogramName, histogram] : other.fIntegerHistograms) {
    auto &target = fIntegerHistograms[histogramName];
    for (const auto &[value, count] : histogram) {
      target[value] += count;
    }
  }

  for (const auto &[histogramName, histogram] : other.fValueHistograms) {
    auto &target = fValueHistograms[histogramName];
    for (const auto &[value, count] : histogram) {
      target[value] += count;
    }
  }

  for (const auto &[volumeId, entry] : other.fEnergyDepositByVolume) {
    auto &target = fEnergyDepositByVolume[volumeId];
    target.label = entry.label;
    target.total += entry.total;
  }
}

void TruthHistogrammer::WriteROOTFile(const std::string &path) const
{
#ifdef ADEPT_INTEGRATIONTEST_HAS_ROOT
  const auto outputPath = std::filesystem::path(path);
  if (outputPath.has_parent_path()) {
    std::filesystem::create_directories(outputPath.parent_path());
  }

  TFile file(path.c_str(), "RECREATE");
  if (file.IsZombie()) {
    throw std::runtime_error("Failed to create ROOT output file: " + path);
  }

  for (const auto &[name, histogram] : fCategoricalHistograms) {
    WriteCountHistogram(file, name, histogram);
  }
  for (const auto &[name, histogram] : fIntegerHistograms) {
    WriteIntegerHistogram(file, name, histogram);
  }
  for (const auto &[name, histogram] : fValueHistograms) {
    WriteValueHistogram(file, name, histogram);
  }
  WriteEnergyDepositHistogram(file, fEnergyDepositByVolume);

  file.Write();
  file.Close();
#else
  (void)path;
  throw std::runtime_error("ROOT truth output requested, but integrationTest was built without ROOT support");
#endif
}
