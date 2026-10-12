// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Validation of the GPU-to-Geant4 handoff of tracks leaving the GPU regions.
///
/// When only some regions are transported on the GPU, a track leaving them is returned to Geant4 with a touchable
/// built from the VecGeom state it reached. Geant4 starts tracking it with
/// G4Navigator::ResetHierarchyAndLocate(position, direction, touchable), a relative search from that path which
/// decides by the direction on surfaces. The handoff is correct when Geant4 confirms the path, or when it differs only
/// for a geometric reason that native Geant4 tracking would also follow. The validator repeats that relocation for
/// every exit with its own navigator and classifies the outcome.
///
/// One validator per Geant4 worker thread: no locking while tracking. At the end of the job each validator merges its
/// counters and records into a process-wide summary, which prints the totals and writes <prefix>_handoff.json when
/// the last validator is destroyed.

#pragma once

#include <G4ThreeVector.hh>
#include <G4TouchableHandle.hh>
#include <G4Types.hh>

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

class G4NavigationHistory;
class G4Navigator;

class HandoffValidator {
public:
  /// Outcome counters (summed over worker threads at the end of the job).
  struct Counts {
    long exits{0};              // tracks leaving the GPU regions
    long confirmed{0};          // Geant4 locates the handed point in the expected path
    long reversal{0};           // relocated back across the crossed surface, as the normal at the exit point says
    long coincident{0};         // relocated into a further volume whose surface also contains the boundary point
    long pushCrossing{0};       // GPU path confirmed at the end point, but the handoff push crossed a further surface
    long bandErrors{0};         // boundary point outside Geant4's tolerance band of the crossing (any reason below)
    long bandExited{0};         // ... strictly inside a volume being left
    long bandEntered{0};        // ... strictly outside a volume being entered
    long bandOff{0};            // ... not on the surface of any crossed volume
    long leaving{0};            // normal at the exit point . direction > 0: leaving through the crossed surface
    long reversed{0};           // ... < 0: the direction points back through it
    long tangent{0};            // ... == 0
    long unexplained{0};        // relocated elsewhere without a geometric reason (overlaps, solid disagreements)
    long bandOverlap{0};        // band errors proven to come from an overlap
    long unexplainedOverlap{0}; // unexplained relocations proven to come from an overlap
    Counts &operator+=(Counts const &other);
    /// True when every tolerance-band violation and unexplained relocation is proven to come from an overlap.
    bool Passed() const { return bandErrors == bandOverlap && unexplained == unexplainedOverlap; }
  };

  /// Proven overlap pair, keyed by "overlap: A and B" / "extrusion: D out of M".
  struct OverlapEntry {
    long count{0};
    double depth{0.};   // deepest evidence, mm
    double witness{0.}; // farthest witness point, signed along the direction from the end point, mm
  };

  /// A handoff not confirmed by Geant4 at the first attempt, with what is needed to replay it.
  struct Record {
    std::string category; // band error, direction reversed, coincident surface, push crossing, unexplained
    std::string band;     // band error detail ("" if none)
    bool provenOverlap{false};
    int event{-1};
    unsigned long long track{0}; // GPU track id (reproducible)
    G4ThreeVector boundary;      // GPU end point
    G4ThreeVector handed;        // position given to Geant4
    G4ThreeVector direction;
    double outwardNDotD{0.}; // normal of the crossed surface . direction, out of the side being left
    // Geant4 paths from the world as (physical volume name, copy number) per level
    std::vector<std::pair<std::string, int>> exited, expected, located;
  };

  /// @param outputPrefix Prefix of <prefix>_handoff.json (empty: no file).
  /// @param recordCapacity Maximum number of records kept.
  HandoffValidator(std::string const &outputPrefix, unsigned int recordCapacity);
  ~HandoffValidator();

  HandoffValidator(const HandoffValidator &)            = delete;
  HandoffValidator &operator=(const HandoffValidator &) = delete;

  /// @brief Check one track leaving the GPU regions.
  /// @param exited Geant4 path of the volume the GPU step started in (the GPU side of the crossing).
  /// @param expected Geant4 path of the VecGeom state reached (the CPU side), given to Geant4 as the touchable.
  /// @param boundaryPoint Global step end point found by the GPU transport, on the crossed surfaces.
  /// @param handedPoint Global position given to Geant4 (the boundary point, possibly pushed along the direction).
  /// @param direction Global direction given to Geant4.
  /// @param event Geant4 event id, @param track GPU track id (for the records).
  void Check(G4TouchableHandle const &exited, G4TouchableHandle const &expected, G4ThreeVector const &boundaryPoint,
             G4ThreeVector const &handedPoint, G4ThreeVector const &direction, int event, unsigned long long track);

  /// @brief Print this worker's counters and verdict.
  void Report(std::ostream &out, int threadId) const;

  bool Passed() const { return fCounts.Passed(); }

private:
  /// Geant4 classification of a point for one level of a path.
  struct LevelView {
    G4int inside;   // EInside of the level's solid at the point
    G4double nDotD; // outward normal of the level's solid . direction, in the solid frame
  };
  LevelView View(G4NavigationHistory const &path, std::size_t level, G4ThreeVector const &point,
                 G4ThreeVector const &direction) const;
  void Detail(std::string const &what, G4ThreeVector const &boundaryPoint, G4ThreeVector const &direction,
              std::string const &volumes);
  /// Prove an overlap from the Geant4 solids of two paths, at `point` or at witness points before or after it along
  /// `direction`: strictly inside volumes of both paths below their deepest common volume (overlap), or strictly
  /// inside a volume and strictly outside one of its ancestors (extrusion). The proven pair is recorded; returns false
  /// without proof.
  bool ProveOverlap(G4NavigationHistory const &a, G4NavigationHistory const &b, G4ThreeVector const &point,
                    G4ThreeVector const &direction);
  bool ProveOverlapAt(G4NavigationHistory const &a, G4NavigationHistory const &b, G4ThreeVector const &point,
                      double witness);
  void AddRecord(Record &&record);
  /// Classify the relocation outcome (counting it): "confirmed", "direction reversed", "coincident surface",
  /// "crossed by handoff push" or "unexplained".
  std::string Classify(G4NavigationHistory const &from, G4NavigationHistory const &to,
                       G4NavigationHistory const &located, std::size_t common, bool onCrossedSurface, double outwardNd,
                       G4ThreeVector const &boundaryPoint, G4ThreeVector const &handedPoint,
                       G4ThreeVector const &direction, G4TouchableHandle const &expected, bool &provenOverlap);

  std::unique_ptr<G4Navigator> fNavigator;
  Counts fCounts;
  std::map<std::string, OverlapEntry> fOverlaps;
  std::vector<Record> fRecords;
  unsigned int fRecordCapacity{0};
  int fDetailsPrinted{0};

  class Summary;
  std::shared_ptr<Summary> fSummary; // process-wide, shared by the validators of all worker threads
};
