// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Validation of the GPU-to-Geant4 handoff (/adept/validateHandoff).
///
/// A track leaving the GPU regions is returned to Geant4 with a touchable built from the VecGeom state it reached,
/// and Geant4 relocates it from that path with G4Navigator::ResetHierarchyAndLocate. The validator repeats that
/// relocation with its own navigator and classifies the outcome. One validator per worker thread, which prints its
/// counts at the end of each run.

#pragma once

#include <G4ThreeVector.hh>
#include <G4TouchableHandle.hh>
#include <G4VStateDependent.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>

class G4NavigationHistory;
class G4Navigator;
class G4TouchableHistory;

enum class HandoffOutcome {
  // Conclusive, correct
  Confirmed,         ///< Geant4 relocates into the path reached on the GPU, at the end point and after the push
  CoincidentSurface, ///< Geant4 goes deeper, into a volume whose surface holds the end point and that the direction
                     ///< enters (native Geant4 would take a zero step into it)
  // Inconclusive: not counted as checked
  PushCrossing, ///< Geant4 accepts the GPU path at the end point, but the 1e-7 mm handoff push crosses the next
                ///< surface: what it skips is not checked
  Overlap,      ///< an overlap or extrusion is proven at the end point or close to it along the direction
  Replicas,     ///< a parameterised or replicated volume in the paths: their solids are not checked
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
  std::uint64_t Total() const;
  std::uint64_t Inconclusive() const;
  std::uint64_t Unexpected() const;
  std::uint64_t Conclusive() const { return Total() - Inconclusive(); }

  enum class Verdict { NoConclusiveHandoff, Passed, Failed };
  /// Failed if any handoff is wrong, else Passed if at least one handoff was conclusive.
  Verdict GetVerdict() const;
  void Print(std::ostream &out) const;
};

class HandoffValidator : public G4VStateDependent {
public:
  HandoffValidator();
  ~HandoffValidator() override;

  /// Classify the handoff of a track that left the GPU regions on a boundary.
  /// @param exited Path of the volume the step was made in (the GPU pre-step state).
  /// @param expected Path reached on the GPU, given to Geant4 with the track.
  /// @param endPoint Step end point found on the GPU; `handedPoint` is where the track is given to Geant4.
  HandoffOutcome Check(G4TouchableHandle const &exited, G4TouchableHandle const &expected,
                       G4ThreeVector const &endPoint, G4ThreeVector const &handedPoint, G4ThreeVector const &direction);

  HandoffCounts const &GetCounts() const { return fCounts; }

  /// Print this thread's counts and verdict, and reset them.
  void EndOfRun(std::ostream &out);

  /// Calls EndOfRun at the end of each run of this thread (GeomClosed -> Idle).
  G4bool Notify(G4ApplicationState requestedState) override;

private:
  HandoffOutcome Classify(G4NavigationHistory const &from, G4NavigationHistory const &to, G4ThreeVector const &endPoint,
                          G4ThreeVector const &handedPoint, G4ThreeVector const &direction,
                          G4TouchableHandle const &expected);
  /// Relocate as Geant4 does from the `start` path; nullptr if the point is outside the world.
  std::unique_ptr<G4TouchableHistory> Locate(G4ThreeVector const &point, G4ThreeVector const &direction,
                                             G4TouchableHandle const &start);
  void PrintUnexpected(HandoffOutcome outcome, G4NavigationHistory const &from, G4NavigationHistory const &to,
                       G4NavigationHistory const *located, G4ThreeVector const &endPoint,
                       G4ThreeVector const &direction);

  std::unique_ptr<G4Navigator> fNavigator;
  HandoffCounts fCounts;
  int fPrinted{0};
};
