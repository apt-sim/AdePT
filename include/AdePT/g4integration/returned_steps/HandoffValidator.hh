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

#pragma once

#include <G4ThreeVector.hh>
#include <G4TouchableHandle.hh>
#include <G4Types.hh>

#include <map>
#include <memory>
#include <ostream>
#include <string>

class G4NavigationHistory;
class G4Navigator;

class HandoffValidator {
public:
  HandoffValidator();
  ~HandoffValidator();

  HandoffValidator(const HandoffValidator &)            = delete;
  HandoffValidator &operator=(const HandoffValidator &) = delete;

  /// @brief Check one track leaving the GPU regions.
  /// @param exited Geant4 path of the volume the GPU step started in (the GPU side of the crossing).
  /// @param expected Geant4 path of the VecGeom state reached (the CPU side), given to Geant4 as the touchable.
  /// @param boundaryPoint Global step end point found by the GPU transport, on the crossed surfaces.
  /// @param handedPoint Global position given to Geant4 (the boundary point, possibly pushed along the direction).
  /// @param direction Global direction given to Geant4.
  void Check(G4TouchableHandle const &exited, G4TouchableHandle const &expected, G4ThreeVector const &boundaryPoint,
             G4ThreeVector const &handedPoint, G4ThreeVector const &direction);

  /// @brief Print the counters and the verdict.
  void Report(std::ostream &out, int threadId) const;

  /// @brief True when every tolerance-band violation and unexplained relocation is proven to come from an overlap.
  /// @details Proven overlaps are geometry errors, listed in the report; they do not fail the transport check.
  bool Passed() const { return fBandErrors == fBandOverlap && fUnexplained == fUnexplainedOverlap; }

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
  /// Prove an overlap from the Geant4 solids of two paths, at `point` or at witness points just beyond it along
  /// `direction`: strictly inside volumes of both paths below their deepest common volume (overlap), or strictly
  /// inside a volume and strictly outside one of its ancestors (extrusion). The proven pair is recorded; returns false
  /// without proof.
  bool ProveOverlap(G4NavigationHistory const &a, G4NavigationHistory const &b, G4ThreeVector const &point,
                    G4ThreeVector const &direction);
  bool ProveOverlapAt(G4NavigationHistory const &a, G4NavigationHistory const &b, G4ThreeVector const &point,
                      double witness);

  struct OverlapEntry {
    long count{0};
    double depth{0.};   // deepest evidence, mm
    double witness{0.}; // largest distance of a witness point beyond the end point, mm
  };
  std::map<std::string, OverlapEntry> fOverlaps; // "overlap: A and B" / "extrusion: D out of M"

  std::unique_ptr<G4Navigator> fNavigator;

  long fExits{0};              // tracks leaving the GPU regions
  long fConfirmed{0};          // Geant4 locates the handed point in the expected path
  long fReversal{0};           // relocated back across the crossed surface, as the normal at the exit point says
  long fCoincident{0};         // relocated into a further volume whose surface also contains the boundary point
  long fBandErrors{0};         // boundary point outside Geant4's tolerance band of the crossing (any reason below)
  long fBandExited{0};         // ... strictly inside a volume being left
  long fBandEntered{0};        // ... strictly outside a volume being entered
  long fBandOff{0};            // ... not on the surface of any crossed volume
  long fLeaving{0};            // normal at the exit point . direction > 0: leaving through the crossed surface
  long fReversed{0};           // ... < 0: the direction points back through it
  long fTangent{0};            // ... == 0
  long fUnexplained{0};        // relocated elsewhere without a geometric reason (overlaps, solid disagreements)
  long fBandOverlap{0};        // band errors proven to come from an overlap
  long fUnexplainedOverlap{0}; // unexplained relocations proven to come from an overlap
  int fDetailsPrinted{0};
};
