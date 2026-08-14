// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <G4NavigationHistory.hh>
#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>

#define private public
#include <AdePT/g4integration/returned_steps/AdePTGeant4Integration.hh>
#undef private

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>

#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <G4GDMLParser.hh>
#include <G4GeometryManager.hh>
#include <G4PVReplica.hh>
#include <G4ReplicaNavigation.hh>
#include <G4TransportationManager.hh>
#include <G4VPVParameterisation.hh>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct NavigationPair {
  vecgeom::NavigationState pre;
  vecgeom::NavigationState post;
};

struct SampleStats {
  std::size_t visitedPlacements = 0;
  std::size_t maxDepth          = 0;
};

G4VPhysicalVolume *MutableG4Volume(AdePTGeometryBridge::MappedVolumeInstance const &instance)
{
  return const_cast<G4VPhysicalVolume *>(instance.g4Volume);
}

void StampG4VolumeInstance(AdePTGeometryBridge::MappedVolumeInstance const &instance)
{
  switch (instance.type) {
  case kReplica: {
    G4ReplicaNavigation nav;
    auto *volume = MutableG4Volume(instance);
    nav.ComputeTransformation(instance.copyNo, volume);
    volume->SetCopyNo(instance.copyNo);
    break;
  }
  case kParameterised: {
    auto *volume           = MutableG4Volume(instance);
    auto *parameterisation = volume->GetParameterisation();
    if (parameterisation == nullptr) {
      throw std::runtime_error("Parameterized Geant4 volume has no parameterisation");
    }
    parameterisation->ComputeTransformation(instance.copyNo, volume);
    volume->SetCopyNo(instance.copyNo);
    break;
  }
  default:
    break;
  }
}

bool MatchesHistoryLevel(G4NavigationHistory const &history, G4int level,
                         AdePTGeometryBridge::MappedVolumeInstance const &instance)
{
  if (history.GetVolume(level) != instance.g4Volume) return false;
  if (level == 0) return true;
  return history.GetVolumeType(level) == instance.type && history.GetReplicaNo(level) == instance.copyNo;
}

// This is the pre-optimization implementation, kept here as an executable
// reference for both exact-equivalence checks and side-by-side perf runs.
[[gnu::noinline]] void ReferenceFillG4NavigationHistory(vecgeom::NavigationState const &navState,
                                                        G4NavigationHistory &history)
{
  auto historyDepth   = history.GetDepth();
  const auto navLevel = navState.GetLevel();

  unsigned int level = 0;
  for (; level <= navLevel; ++level) {
    auto const *placedVolume = navState.At(static_cast<int>(level));
    auto *g4Volume           = AdePTGeometryBridge::GetG4PhysicalVolume(placedVolume);
    const auto type          = g4Volume->VolumeType();
    const auto copyNo        = type == kNormal ? g4Volume->GetCopyNo() : placedVolume->GetCopyNo();
    const AdePTGeometryBridge::MappedVolumeInstance newInstance{g4Volume, type, copyNo};

    if (historyDepth && level <= historyDepth) {
      if (MatchesHistoryLevel(history, static_cast<G4int>(level), newInstance)) {
        if (level) StampG4VolumeInstance(newInstance);
        continue;
      }
      if (level) {
        history.BackLevel(static_cast<G4int>(historyDepth - level + 1));
        StampG4VolumeInstance(newInstance);
        history.NewLevel(MutableG4Volume(newInstance), newInstance.type, newInstance.copyNo);
      } else {
        history.BackLevel(static_cast<G4int>(historyDepth));
        history.SetFirstEntry(MutableG4Volume(newInstance));
      }
      historyDepth = level;
    } else {
      if (level) {
        StampG4VolumeInstance(newInstance);
        history.NewLevel(MutableG4Volume(newInstance), newInstance.type, newInstance.copyNo);
        ++historyDepth;
      } else {
        history.SetFirstEntry(MutableG4Volume(newInstance));
      }
    }
  }
  if (historyDepth >= level) history.BackLevel(static_cast<G4int>(historyDepth - level + 1));
}

[[gnu::noinline]] G4TouchableHandle ReferenceMakeTouchable(vecgeom::NavigationState const &navState)
{
  auto navigationHistory = std::make_unique<G4NavigationHistory>();
  ReferenceFillG4NavigationHistory(navState, *navigationHistory);
  return G4TouchableHandle(new G4TouchableHistory(*navigationHistory));
}

bool HistoriesAreExactlyEqual(G4NavigationHistory const &actual, G4NavigationHistory const &expected)
{
  if (actual.GetDepth() != expected.GetDepth()) return false;
  for (std::size_t level = 0; level <= actual.GetDepth(); ++level) {
    const auto g4Level = static_cast<G4int>(level);
    if (actual.GetVolume(g4Level) != expected.GetVolume(g4Level) ||
        actual.GetVolumeType(g4Level) != expected.GetVolumeType(g4Level) ||
        actual.GetReplicaNo(g4Level) != expected.GetReplicaNo(g4Level)) {
      return false;
    }

    auto const &actualTransform   = actual.GetTransform(g4Level);
    auto const &expectedTransform = expected.GetTransform(g4Level);
    for (int row = 0; row < 3; ++row) {
      if (actualTransform.NetTranslation()[row] != expectedTransform.NetTranslation()[row]) return false;
      for (int column = 0; column < 3; ++column) {
        if (actualTransform.NetRotation()(row, column) != expectedTransform.NetRotation()(row, column)) return false;
      }
    }
  }
  return true;
}

void CollectNavigationPairs(vecgeom::VPlacedVolume const *placed, vecgeom::NavigationState const &state,
                            std::vector<NavigationPair> &pairs, std::size_t maxPairs, SampleStats &stats)
{
  if (pairs.size() >= maxPairs) return;

  ++stats.visitedPlacements;
  stats.maxDepth        = std::max(stats.maxDepth, static_cast<std::size_t>(state.GetLevel()));
  auto const &daughters = placed->GetLogicalVolume()->GetDaughters();

  for (std::size_t childIndex = 0; childIndex < daughters.size() && pairs.size() < maxPairs; ++childIndex) {
    auto childState = state;
    childState.PushDaughter(static_cast<int>(childIndex));
    if (childState.Top() != daughters[childIndex]) {
      throw std::runtime_error("Synthetic navigation state does not match the VecGeom placement tree");
    }

    // Four same-volume steps per boundary step approximate the usual dominance
    // of steps whose pre/post histories are identical. The boundary pair still
    // shares the entire parent prefix.
    for (int repeat = 0; repeat < 4 && pairs.size() < maxPairs; ++repeat) {
      pairs.push_back({childState, childState});
    }
    if (pairs.size() < maxPairs) pairs.push_back({state, childState});

    CollectNavigationPairs(daughters[childIndex], childState, pairs, maxPairs, stats);
  }
}

void VerifyExactEquivalence(AdePTGeant4Integration &integration, std::vector<NavigationPair> const &pairs)
{
  G4NavigationHistory referencePre;
  G4NavigationHistory referencePost;
  G4NavigationHistory optimizedPre;
  G4NavigationHistory optimizedPost;

  for (std::size_t index = 0; index < pairs.size(); ++index) {
    auto const &pair = pairs[index];
    ReferenceFillG4NavigationHistory(pair.pre, referencePre);
    ReferenceFillG4NavigationHistory(pair.post, referencePost);
    integration.FillG4NavigationHistories(pair.pre, pair.post, optimizedPre, optimizedPost);

    if (!HistoriesAreExactlyEqual(optimizedPre, referencePre) ||
        !HistoriesAreExactlyEqual(optimizedPost, referencePost)) {
      throw std::runtime_error("Exact history equivalence failed at synthetic step " + std::to_string(index));
    }
  }

  const auto touchableChecks = std::min<std::size_t>(pairs.size(), 4096);
  for (std::size_t index = 0; index < touchableChecks; ++index) {
    auto reference = ReferenceMakeTouchable(pairs[index].post);
    auto optimized = integration.MakeTouchableFromNavState(pairs[index].post);
    if (reference->GetHistoryDepth() != optimized->GetHistoryDepth()) {
      throw std::runtime_error("Touchable depth equivalence failed at synthetic step " + std::to_string(index));
    }
    for (int depth = 0; depth <= reference->GetHistoryDepth(); ++depth) {
      // Geant4 returns thread-local scratch objects for nonzero-depth transforms,
      // so copy each result before making the next accessor call.
      const G4ThreeVector referenceTranslation = reference->GetTranslation(depth);
      const G4RotationMatrix referenceRotation = *reference->GetRotation(depth);
      const G4ThreeVector optimizedTranslation = optimized->GetTranslation(depth);
      const G4RotationMatrix optimizedRotation = *optimized->GetRotation(depth);
      if (reference->GetVolume(depth) != optimized->GetVolume(depth) ||
          reference->GetCopyNumber(depth) != optimized->GetCopyNumber(depth) ||
          referenceTranslation != optimizedTranslation || referenceRotation != optimizedRotation) {
        throw std::runtime_error("Touchable equivalence failed at synthetic step " + std::to_string(index));
      }
    }
  }
}

std::uintptr_t RunPairBenchmark(AdePTGeant4Integration &integration, std::vector<NavigationPair> const &pairs,
                                std::size_t operations, bool optimized)
{
  G4NavigationHistory preHistory;
  G4NavigationHistory postHistory;
  std::uintptr_t checksum = 0;
  for (std::size_t operation = 0; operation < operations; ++operation) {
    auto const &pair = pairs[operation % pairs.size()];
    if (optimized) {
      integration.FillG4NavigationHistories(pair.pre, pair.post, preHistory, postHistory);
    } else {
      ReferenceFillG4NavigationHistory(pair.pre, preHistory);
      ReferenceFillG4NavigationHistory(pair.post, postHistory);
    }
    checksum += reinterpret_cast<std::uintptr_t>(preHistory.GetTopVolume());
    checksum += static_cast<std::uintptr_t>(postHistory.GetTopReplicaNo() + 1);
  }
  return checksum;
}

std::uintptr_t RunTouchableBenchmark(AdePTGeant4Integration &integration, std::vector<NavigationPair> const &pairs,
                                     std::size_t operations, bool optimized)
{
  std::uintptr_t checksum = 0;
  for (std::size_t operation = 0; operation < operations; ++operation) {
    auto const &navState = pairs[operation % pairs.size()].post;
    auto touchable = optimized ? integration.MakeTouchableFromNavState(navState) : ReferenceMakeTouchable(navState);
    checksum += reinterpret_cast<std::uintptr_t>(touchable->GetVolume());
    checksum += static_cast<std::uintptr_t>(touchable->GetCopyNumber() + 1);
  }
  return checksum;
}

std::size_t ParseSize(char const *text, char const *name)
{
  char *end         = nullptr;
  const auto result = std::strtoull(text, &end, 10);
  if (text == end || *end != '\0' || result == 0) throw std::runtime_error(std::string("Invalid ") + name);
  return static_cast<std::size_t>(result);
}

} // namespace

int main(int argc, char **argv)
try {
  if (argc < 3 || argc > 6) {
    std::cerr << "Usage: " << argv[0]
              << " <geometry.gdml> <reference-pair|optimized-pair|reference-touchable|optimized-touchable>"
                 " [operations] [sample-pairs] [--skip-verify]\n";
    return 2;
  }

  const std::string geometryFile = argv[1];
  const std::string_view mode    = argv[2];
  const std::size_t operations   = argc >= 4 ? ParseSize(argv[3], "operation count") : 10'000'000;
  const std::size_t maxPairs     = argc >= 5 ? ParseSize(argv[4], "sample-pair count") : 65'536;
  const bool skipVerify          = argc == 6 && std::string_view(argv[5]) == "--skip-verify";
  if (argc == 6 && !skipVerify) throw std::runtime_error("Unknown final argument");

  const bool optimized = mode == "optimized-pair" || mode == "optimized-touchable";
  const bool pairMode  = mode == "reference-pair" || mode == "optimized-pair";
  const bool touchMode = mode == "reference-touchable" || mode == "optimized-touchable";
  if (!pairMode && !touchMode) throw std::runtime_error("Unknown benchmark mode");

  G4GDMLParser parser;
  parser.Read(geometryFile, false);
  auto *world = parser.GetWorldVolume();
  if (world == nullptr) throw std::runtime_error("GDML parser returned a null world");

  G4TransportationManager::GetTransportationManager()->SetWorldForTracking(world);
  G4GeometryManager::GetInstance()->CloseGeometry(true);
  AdePTGeometryBridge::CreateVecGeomWorld(world);

  auto const *vecgeomWorld = vecgeom::GeoManager::Instance().GetWorld();
  if (vecgeomWorld == nullptr) throw std::runtime_error("VecGeom world is null");

  vecgeom::NavigationState worldState;
  worldState.Push(vecgeomWorld);
  std::vector<NavigationPair> pairs;
  pairs.reserve(maxPairs);
  SampleStats stats;
  CollectNavigationPairs(vecgeomWorld, worldState, pairs, maxPairs, stats);
  if (pairs.empty()) throw std::runtime_error("Geometry did not produce any navigation pairs");

  AdePTGeant4Integration integration;
  if (!skipVerify) VerifyExactEquivalence(integration, pairs);

  const auto warmupOperations = std::min<std::size_t>(operations, 100'000);
  if (pairMode) {
    RunPairBenchmark(integration, pairs, warmupOperations, optimized);
  } else {
    RunTouchableBenchmark(integration, pairs, warmupOperations, optimized);
  }

  const auto start    = Clock::now();
  const auto checksum = pairMode ? RunPairBenchmark(integration, pairs, operations, optimized)
                                 : RunTouchableBenchmark(integration, pairs, operations, optimized);
  const auto elapsed  = std::chrono::duration<double>(Clock::now() - start).count();

  std::cout << "mode=" << mode << " operations=" << operations << " sample_pairs=" << pairs.size()
            << " visited_placements=" << stats.visitedPlacements << " max_depth=" << stats.maxDepth
            << " verified=" << (!skipVerify) << " elapsed_seconds=" << elapsed
            << " ns_per_operation=" << elapsed * 1.e9 / static_cast<double>(operations) << " checksum=" << checksum
            << '\n';
  return 0;
} catch (std::exception const &error) {
  std::cerr << "benchmark_navigation_history: " << error.what() << '\n';
  return 1;
}
