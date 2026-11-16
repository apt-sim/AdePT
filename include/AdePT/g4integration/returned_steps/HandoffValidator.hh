// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Validation of the GPU-to-Geant4 handoff (/adept/validateHandoff).
///
/// A track leaving the GPU regions is returned to Geant4 with a touchable built from the VecGeom state it reached,
/// and Geant4 relocates it from that path with G4Navigator::ResetHierarchyAndLocate. The validator repeats that
/// relocation with its own navigator and classifies the outcome. One validator per worker thread; the counts of all
/// threads are summed at the end of each run, and the last thread prints them.

#pragma once

#include <G4ThreeVector.hh>
#include <G4TouchableHandle.hh>
#include <G4VStateDependent.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ostream>

class G4NavigationHistory;
class G4Navigator;

enum class HandoffOutcome {
  // Conclusive, correct
  Confirmed,         ///< Geant4 relocates into the path reached on the GPU
  CoincidentSurface, ///< Geant4 goes deeper, into a volume whose surface holds the end point and that the direction
                     ///< enters (native Geant4 would take a zero step into it)
  PushCrossing,      ///< Geant4 confirms the GPU path at the end point; the handoff push crossed the next surface
  // Inconclusive: not counted as checked
  Overlap,  ///< an overlap or extrusion is proven at the end point or close to it along the direction
  Replicas, ///< a parameterised or replicated volume in the paths: their solids are not checked
  // Conclusive, wrong
  BandError,         ///< end point beyond Geant4's tolerance band of the crossed surfaces
  DirectionReversed, ///< the final direction points back through the crossed surface: the GPU turn-back check was
                     ///< skipped, or its VecGeom normal disagrees with Geant4's
  OtherPath,         ///< Geant4 relocates elsewhere, without a geometric reason
  Count
};

char const *HandoffOutcomeName(HandoffOutcome outcome);

/// Outcome counts of one run.
struct HandoffCounts {
  std::array<std::uint64_t, std::size_t(HandoffOutcome::Count)> n{};

  std::uint64_t &operator[](HandoffOutcome outcome) { return n[std::size_t(outcome)]; }
  std::uint64_t operator[](HandoffOutcome outcome) const { return n[std::size_t(outcome)]; }
  HandoffCounts &operator+=(HandoffCounts const &other);
  std::uint64_t Total() const;
  std::uint64_t Inconclusive() const { return (*this)[HandoffOutcome::Overlap] + (*this)[HandoffOutcome::Replicas]; }
  std::uint64_t Unexpected() const;
  std::uint64_t Conclusive() const { return Total() - Inconclusive(); }

  enum class Verdict { NoConclusiveHandoff, Passed, Failed };
  /// Failed if any handoff is wrong, else Passed if at least one handoff was conclusive.
  Verdict GetVerdict() const;
  void Print(std::ostream &out, int threads) const;
};

/// Sums the counts of the worker threads for one run.
class HandoffRunTotals {
public:
  /// Adds the counts of one thread. When all `threads` have added theirs, returns true with their sum in `total`
  /// and starts over for the next run.
  bool Add(HandoffCounts const &counts, int threads, HandoffCounts &total);

private:
  std::mutex fMutex;
  HandoffCounts fSum;
  int fAdded{0};
};

class HandoffValidator : public G4VStateDependent {
public:
  /// @param threads Number of worker threads whose counts are summed at the end of a run.
  explicit HandoffValidator(int threads);
  ~HandoffValidator() override;

  /// Classify the handoff of a track that left the GPU regions on a boundary.
  /// @param exited Path of the volume the step was made in (the GPU pre-step state).
  /// @param expected Path reached on the GPU, given to Geant4 with the track.
  /// @param endPoint Step end point found on the GPU; `handedPoint` is where the track is given to Geant4.
  HandoffOutcome Check(G4TouchableHandle const &exited, G4TouchableHandle const &expected,
                       G4ThreeVector const &endPoint, G4ThreeVector const &handedPoint, G4ThreeVector const &direction);

  HandoffCounts const &GetCounts() const { return fCounts; }

  /// Add this thread's counts to `totals`, print them if this is the last thread, and reset the counts.
  void EndOfRun(HandoffRunTotals &totals, std::ostream &out);

  /// Calls EndOfRun with the process-wide totals at the end of each run (GeomClosed -> Idle).
  G4bool Notify(G4ApplicationState requestedState) override;

private:
  HandoffOutcome Classify(G4NavigationHistory const &from, G4NavigationHistory const &to, G4ThreeVector const &endPoint,
                          G4ThreeVector const &handedPoint, G4ThreeVector const &direction,
                          G4TouchableHandle const &expected);
  void PrintUnexpected(HandoffOutcome outcome, G4NavigationHistory const &from, G4NavigationHistory const &to,
                       G4NavigationHistory const &located, G4ThreeVector const &endPoint,
                       G4ThreeVector const &direction);

  std::unique_ptr<G4Navigator> fNavigator;
  HandoffCounts fCounts;
  int fThreads;
  int fPrinted{0};
};
