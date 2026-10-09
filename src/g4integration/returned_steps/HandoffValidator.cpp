// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/returned_steps/HandoffValidator.hh>

#include <G4AffineTransform.hh>
#include <G4GeometryTolerance.hh>
#include <G4LogicalVolume.hh>
#include <G4NavigationHistory.hh>
#include <G4Navigator.hh>
#include <G4TouchableHistory.hh>
#include <G4TransportationManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSolid.hh>

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace {

constexpr int kMaxDetails = 20;

bool SameLevel(G4NavigationHistory const &a, G4NavigationHistory const &b, std::size_t level)
{
  return a.GetVolume(level) == b.GetVolume(level) && a.GetReplicaNo(level) == b.GetReplicaNo(level);
}

bool SamePath(G4NavigationHistory const &a, G4NavigationHistory const &b)
{
  if (a.GetDepth() != b.GetDepth()) return false;
  for (std::size_t i = 0; i <= a.GetDepth(); ++i)
    if (!SameLevel(a, b, i)) return false;
  return true;
}

/// True if `path` passes through level `level` of `reference` (same volumes down to that level).
bool Contains(G4NavigationHistory const &path, G4NavigationHistory const &reference, std::size_t level)
{
  if (path.GetDepth() < level) return false;
  for (std::size_t i = 0; i <= level; ++i)
    if (!SameLevel(path, reference, i)) return false;
  return true;
}

std::string Name(G4NavigationHistory const &path, std::size_t level)
{
  return path.GetVolume(level) ? std::string(path.GetVolume(level)->GetName()) : std::string("none");
}

} // namespace

HandoffValidator::HandoffValidator() : fNavigator(std::make_unique<G4Navigator>())
{
  fNavigator->SetWorldVolume(
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());
}

HandoffValidator::~HandoffValidator() = default;

HandoffValidator::LevelView HandoffValidator::View(G4NavigationHistory const &path, std::size_t level,
                                                   G4ThreeVector const &point, G4ThreeVector const &direction) const
{
  // Transform of `level` within the path: the top transform of the truncated history.
  G4AffineTransform toLocal = path.GetTransform(level);
  auto const *solid         = path.GetVolume(level)->GetLogicalVolume()->GetSolid();
  const G4ThreeVector local = toLocal.TransformPoint(point);
  const auto inside         = solid->Inside(local);
  const double nDotD = inside == kSurface ? solid->SurfaceNormal(local).dot(toLocal.TransformAxis(direction)) : 0.;
  return {G4int(inside), nDotD};
}

bool HandoffValidator::ProveOverlap(G4NavigationHistory const &a, G4NavigationHistory const &b,
                                    G4ThreeVector const &point, G4ThreeVector const &direction)
{
  const double tolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
  for (double witness : {0., 2., 10., 100.})
    if (ProveOverlapAt(a, b, point + witness * tolerance * direction, witness * tolerance)) return true;
  return false;
}

bool HandoffValidator::ProveOverlapAt(G4NavigationHistory const &a, G4NavigationHistory const &b,
                                      G4ThreeVector const &point, double witness)
{
  // Strict Geant4 distance of the point to the solid of `level`: > 0 strictly inside, < 0 strictly outside, 0 on the
  // surface (within Geant4's band).
  auto strict = [&](G4NavigationHistory const &path, std::size_t level) {
    G4AffineTransform toLocal = path.GetTransform(level);
    auto const *solid         = path.GetVolume(level)->GetLogicalVolume()->GetSolid();
    const G4ThreeVector local = toLocal.TransformPoint(point);
    const auto inside         = solid->Inside(local);
    return inside == kInside ? solid->DistanceToOut(local) : inside == kOutside ? -solid->DistanceToIn(local) : 0.;
  };
  auto name = [](G4NavigationHistory const &path, std::size_t level) {
    return Name(path, level) + "#" + std::to_string(path.GetReplicaNo(level));
  };
  auto record = [&](std::string const &key, double depth) {
    auto &entry = fOverlaps[key];
    ++entry.count;
    entry.depth   = std::max(entry.depth, depth);
    entry.witness = std::max(entry.witness, witness);
    return true;
  };

  std::size_t common = 0;
  while (common < a.GetDepth() && common < b.GetDepth() && SameLevel(a, b, common + 1))
    ++common;

  // Deepest volume of each path below the common volume that strictly contains the point.
  std::size_t insideA = 0, insideB = 0;
  double depthA = 0., depthB = 0.;
  for (std::size_t level = common + 1; level <= a.GetDepth(); ++level)
    if (const double d = strict(a, level); d > 0.) insideA = level, depthA = d;
  for (std::size_t level = common + 1; level <= b.GetDepth(); ++level)
    if (const double d = strict(b, level); d > 0.) insideB = level, depthB = d;
  if (insideA > 0 && insideB > 0)
    return record("overlap: " + name(a, insideA) + " and " + name(b, insideB), std::min(depthA, depthB));

  // Extrusion: a volume strictly containing the point under an ancestor strictly excluding it.
  for (auto const *path : {&a, &b}) {
    const std::size_t inside = path == &a ? insideA : insideB;
    if (inside == 0) continue;
    for (std::size_t level = 1; level < inside; ++level)
      if (const double d = strict(*path, level); d < 0.)
        return record("extrusion: " + name(*path, inside) + " out of " + name(*path, level),
                      std::min(path == &a ? depthA : depthB, -d));
  }
  return false;
}

void HandoffValidator::Detail(std::string const &what, G4ThreeVector const &boundaryPoint,
                              G4ThreeVector const &direction, std::string const &volumes)
{
  if (fDetailsPrinted++ >= kMaxDetails) return;
  std::printf("HandoffValidation ERROR %s: %s; boundary point (%.17g, %.17g, %.17g) direction (%.17g, %.17g, %.17g)\n",
              what.c_str(), volumes.c_str(), boundaryPoint.x(), boundaryPoint.y(), boundaryPoint.z(), direction.x(),
              direction.y(), direction.z());
}

void HandoffValidator::Check(G4TouchableHandle const &exited, G4TouchableHandle const &expected,
                             G4ThreeVector const &boundaryPoint, G4ThreeVector const &handedPoint,
                             G4ThreeVector const &direction)
{
  auto const &from = *exited->GetHistory();
  auto const &to   = *expected->GetHistory();
  ++fExits;

  // Levels below the deepest common volume are crossed: those of `from` are left, those of `to` are entered.
  std::size_t common = 0;
  while (common < from.GetDepth() && common < to.GetDepth() && SameLevel(from, to, common + 1))
    ++common;

  // 1. Tolerance band, with Geant4's own classification (band +-kCarTolerance/2) at the step end point: it must not
  //    be strictly inside a volume being left, nor strictly outside a volume being entered, and it must lie on the
  //    surface of at least one crossed volume (the surface the GPU transport stopped on).
  bool bandError = false;
  // Crossed surface: the deepest left volume, else the outermost entered volume, whose surface holds the point.
  // `outwardNd` is the normal of that surface . direction, oriented out of the side being left.
  bool onCrossedSurface = false;
  double outwardNd      = 0.;
  for (std::size_t level = from.GetDepth(); level > common; --level) {
    const auto view = View(from, level, boundaryPoint, direction);
    if (view.inside == kInside && !bandError) {
      bandError = true;
      ++fBandExited;
      Detail("boundary point inside the volume being left", boundaryPoint, direction,
             "left " + Name(from, level) + ", reached " + Name(to, to.GetDepth()));
    }
    if (view.inside == kSurface && !onCrossedSurface) {
      onCrossedSurface = true;
      outwardNd        = view.nDotD;
    }
  }
  for (std::size_t level = common + 1; level <= to.GetDepth(); ++level) {
    const auto view = View(to, level, boundaryPoint, direction);
    if (view.inside == kOutside && !bandError) {
      bandError = true;
      ++fBandEntered;
      Detail("boundary point outside the volume being entered", boundaryPoint, direction,
             "left " + Name(from, from.GetDepth()) + ", entered " + Name(to, level));
    }
    if (view.inside == kSurface && !onCrossedSurface) {
      onCrossedSurface = true;
      outwardNd        = -view.nDotD;
    }
  }
  if (!onCrossedSurface && !bandError) {
    bandError = true;
    ++fBandOff;
    Detail("boundary point not on the surface of any crossed volume", boundaryPoint, direction,
           "left " + Name(from, from.GetDepth()) + ", reached " + Name(to, to.GetDepth()));
  }
  if (bandError) {
    ++fBandErrors;
    if (ProveOverlap(from, to, boundaryPoint, direction)) ++fBandOverlap;
  }

  // 2. Normal at the exit point: does the final direction leave through the crossed surface?
  if (onCrossedSurface) {
    if (outwardNd > 0.)
      ++fLeaving;
    else if (outwardNd < 0.)
      ++fReversed;
    else
      ++fTangent;
  }

  // 3. Relocation, as G4SteppingManager::SetInitialStep does for a track carrying a touchable.
  fNavigator->ResetHierarchyAndLocate(handedPoint, direction, *static_cast<G4TouchableHistory *>(expected()));
  std::unique_ptr<G4TouchableHistory> locatedTouchable(fNavigator->CreateTouchableHistory());
  auto const &located = *locatedTouchable->GetHistory();
  if (SamePath(located, to)) {
    ++fConfirmed;
    return;
  }

  // Direction reversal: the normal at the exit point says the final direction goes back through the crossed surface
  // (e.g. a deflection applied at the boundary). Native Geant4 tracking would continue on that side after a zero
  // step, so the located path must be on the side being left: inside a left volume, or not inside an entered one.
  if (onCrossedSurface && outwardNd < 0.) {
    bool onLeftSide = false;
    for (std::size_t level = common + 1; !onLeftSide && level <= from.GetDepth(); ++level)
      onLeftSide = View(from, level, boundaryPoint, direction).inside == kSurface && Contains(located, from, level);
    for (std::size_t level = common + 1; !onLeftSide && level <= to.GetDepth(); ++level)
      onLeftSide = View(to, level, boundaryPoint, direction).inside == kSurface && !Contains(located, to, level);
    if (onLeftSide) {
      ++fReversal;
      return;
    }
  }

  // Coincident surface: the located path enters a further volume whose surface also holds the boundary point, with
  // the direction going into it (a zero step for the GPU navigation).
  if (Contains(located, to, to.GetDepth()) && located.GetDepth() > to.GetDepth()) {
    const auto first = View(located, to.GetDepth() + 1, boundaryPoint, direction);
    if (first.inside == kSurface && first.nDotD < 0.) {
      ++fCoincident;
      return;
    }
  }

  ++fUnexplained;
  if (ProveOverlap(located, to, handedPoint, direction) || ProveOverlap(from, to, boundaryPoint, direction)) {
    ++fUnexplainedOverlap;
    return;
  }
  std::ostringstream volumes;
  volumes << "left " << Name(from, from.GetDepth()) << ", reached " << Name(to, to.GetDepth()) << " (depth "
          << to.GetDepth() << "), Geant4 locates " << Name(located, located.GetDepth()) << " (depth "
          << located.GetDepth() << "), normal . direction at the exit point " << outwardNd;
  Detail("relocation not explained by the geometry (overlap or solid disagreement?)", boundaryPoint, direction,
         volumes.str());
}

void HandoffValidator::Report(std::ostream &out, int threadId) const
{
  out << "HandoffValidation thread " << threadId << ": " << fExits << " tracks left the GPU regions\n"
      << "HandoffValidation   tolerance band: " << fBandErrors << " boundary points beyond Geant4's band ("
      << fBandExited << " inside a volume being left, " << fBandEntered << " outside a volume being entered, "
      << fBandOff << " not on any crossed surface)\n"
      << "HandoffValidation   normal at the exit point: " << fLeaving << " leaving through the crossed surface, "
      << fReversed << " direction pointing back, " << fTangent << " tangent\n"
      << "HandoffValidation   relocation: " << fConfirmed << " confirmed, " << fReversal
      << " direction reversed at the boundary, " << fCoincident << " coincident surface, " << fUnexplained
      << " unexplained\n"
      << "HandoffValidation   proven geometry overlaps: " << fBandOverlap << " of the band errors, "
      << fUnexplainedOverlap << " of the unexplained relocations\n";
  for (auto const &[pair, entry] : fOverlaps)
    out << "HandoffValidation   proven " << pair << " (" << entry.count << " tracks, evidence up to " << entry.depth
        << " mm, witness up to " << entry.witness << " mm past the end point)\n";
  out << "HandoffValidation: " << (Passed() ? "PASSED" : "FAILED") << std::endl;
}
