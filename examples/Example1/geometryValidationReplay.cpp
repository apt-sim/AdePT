// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

/// Replay of the records written by the AdePT geometry validation (/adept/geometryValidationOutput).
///
/// Loads a GDML geometry as example1 does (Geant4 GDML reader, then the AdePT geometry bridge to VecGeom), reads the
/// records prepared by replay_geometry_validation.py from standard input and replays each of them on the CPU:
///  C (GPU crossing): rebuilds the VecGeom states, classifies the end point with every crossed solid, recomputes the
///    step from the pre-step point (straight steps) and the relocation with the final direction;
///  H (handoff): rebuilds the Geant4 paths and repeats the Geant4 relocation at the boundary and handed points.
///
/// Usage: geometryValidationReplay <file.gdml> < records.txt

#include <AdePT/g4integration/geometry/AdePTGeometryBridge.hh>

#include <VecGeom/management/BVHManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NavigationState.h>

#include <G4AffineTransform.hh>
#include <G4GDMLParser.hh>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4NavigationHistory.hh>
#include <G4Navigator.hh>
#include <G4TouchableHistory.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSolid.hh>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Vec                   = vecgeom::Vector3D<double>;
constexpr double kPushStuck = 100 * vecgeom::kTolerance; // as in the transport kernels

Vec ReadVec(std::istream &in)
{
  double x, y, z;
  in >> x >> y >> z;
  return {x, y, z};
}

const char *InsideName(int inside)
{
  return inside == vecgeom::EInside::kInside ? "inside" : inside == vecgeom::EInside::kSurface ? "surface" : "outside";
}

/// Rebuild a VecGeom state from daughter indices; checks the labels.
bool ReadState(std::istream &in, vecgeom::NavigationState &state, std::string &error)
{
  int n;
  in >> n;
  state.Clear();
  vecgeom::VPlacedVolume const *volume = vecgeom::GeoManager::Instance().GetWorld();
  for (int level = 0; level < n; ++level) {
    int index;
    std::string label;
    in >> index >> label;
    if (level > 0) {
      auto const &daughters = volume->GetLogicalVolume()->GetDaughters();
      if (index < 0 || index >= int(daughters.size())) {
        error = "daughter index out of range at level " + std::to_string(level);
        return false;
      }
      volume = daughters[index];
    }
    if (volume->GetLabel() != label) {
      error = "label mismatch at level " + std::to_string(level) + ": " + volume->GetLabel() + " != " + label;
      return false;
    }
    state.Push(volume);
  }
  return true;
}

bool SameState(vecgeom::NavigationState const &a, vecgeom::NavigationState const &b)
{
  if (a.GetLevel() != b.GetLevel()) return false;
  for (int level = 0; level <= a.GetLevel(); ++level)
    if (a.At(level) != b.At(level)) return false;
  return true;
}

/// Classification of a global point and direction by one level of a VecGeom state, distances in kTolerance units.
void PrintLevel(vecgeom::NavigationState const &state, int level, Vec const &point, Vec const &dir, char const *role)
{
  vecgeom::Transformation3D m;
  state.TopMatrix(level, m);
  auto const *pv    = state.At(level);
  auto const *solid = pv->GetLogicalVolume()->GetUnplacedVolume();
  const Vec local   = m.Transform(point);
  const Vec ldir    = m.TransformDirection(dir);
  const auto inside = solid->Inside(local);
  Vec normal;
  const bool valid = solid->Normal(local, normal);
  const double k   = vecgeom::kTolerance;
  std::printf("    %-8s level %2d %-40s Inside %-7s SafetyToIn %10.3g SafetyToOut %10.3g DistanceToIn %10.3g "
              "DistanceToOut %10.3g n.d %s%.4g\n",
              role, level, pv->GetLabel().c_str(), InsideName(inside), solid->SafetyToIn(local) / k,
              solid->SafetyToOut(local) / k, solid->DistanceToIn(local, ldir) / k,
              solid->DistanceToOut(local, ldir) / k, valid ? "" : "(invalid) ", normal.Dot(ldir));
}

void ReplayCrossing(std::istream &in, std::string const &id)
{
  int onBoundary, stuck, field;
  vecgeom::NavigationState from, to;
  std::string error;
  in >> onBoundary;
  if (!ReadState(in, from, error) || !ReadState(in, to, error)) {
    std::printf("C %s: cannot rebuild the states (%s)\n", id.c_str(), error.c_str());
    return;
  }
  const Vec prePoint = ReadVec(in), preDir = ReadVec(in), point = ReadVec(in), dir = ReadVec(in);
  in >> stuck >> field;
  from.SetBoundaryState(onBoundary);

  std::printf("C %s: %s -> %s, step %.6g mm%s%s\n", id.c_str(), from.Top()->GetLabel().c_str(),
              to.Top()->GetLabel().c_str(), (point - prePoint).Mag(), stuck ? ", stuck-track push" : "",
              field ? ", magnetic field" : "");
  int common = -1;
  for (int level = 0; level <= std::min(from.GetLevel(), to.GetLevel()) && from.At(level) == to.At(level); ++level)
    common = level;
  std::printf("  end point, final direction (distances in kTolerance):\n");
  for (int level = from.GetLevel(); level > common; --level)
    PrintLevel(from, level, point, dir, "left");
  for (int level = common + 1; level <= to.GetLevel(); ++level)
    PrintLevel(to, level, point, dir, "entered");
  if (common >= 0) PrintLevel(from, common, point, dir, "common");

  std::printf("  pre-step point, pre-step direction (distances in kTolerance):\n");
  for (int level = from.GetLevel(); level >= 0 && level >= common; --level)
    PrintLevel(from, level, prePoint, preDir, level > common ? "pre" : "pre/cmn");
  if (from.IsOnBoundary()) {
    // The BVH navigator computes a step from the boundary at the point pushed by its kBoundaryPush.
    std::printf("  pre-step point pushed by the navigator boundary push (%g kTolerance):\n",
                vecgeom::BVHNavigator::kBoundaryPush / vecgeom::kTolerance);
    const Vec pushed = prePoint + vecgeom::BVHNavigator::kBoundaryPush * preDir;
    PrintLevel(from, from.GetLevel(), pushed, preDir, "pushed");
    for (int level = common + 1; level <= to.GetLevel(); ++level) {
      PrintLevel(to, level, pushed, preDir, "pushed>");
      if (level != common + 1) continue;
      vecgeom::Transformation3D mt;
      to.TopMatrix(level, mt);
      const Vec tp = mt.Transform(pushed), td = mt.TransformDirection(preDir);
      std::printf("    entered local point (%.17g, %.17g, %.17g) direction (%.17g, %.17g, %.17g)\n", tp.x(), tp.y(),
                  tp.z(), td.x(), td.y(), td.z());
      to.At(level)->GetLogicalVolume()->GetUnplacedVolume()->Print();
      std::printf("\n");
    }
    vecgeom::Transformation3D m;
    from.TopMatrix(from.GetLevel(), m);
    const Vec lp = m.Transform(pushed), ld = m.TransformDirection(preDir);
    std::printf("    local point (%.17g, %.17g, %.17g) direction (%.17g, %.17g, %.17g)\n", lp.x(), lp.y(), lp.z(),
                ld.x(), ld.y(), ld.z());
  }

  // Straight step from the pre-step point, as the transport did (with the pre-step direction).
  if (!field) {
    vecgeom::NavigationState next;
    const double step =
        vecgeom::BVHNavigator::ComputeStepAndNextVolume(prePoint, preDir, vecgeom::kInfLength, from, next);
    const Vec end = prePoint + (step + (stuck ? kPushStuck : 0.)) * preDir;
    std::printf("  recomputed step %.9g mm, end point %.3g kTolerance from the recorded one, next %s%s\n", step,
                (end - point).Mag() / vecgeom::kTolerance, next.Top() ? next.Top()->GetLabel().c_str() : "outside",
                next.IsOnBoundary() ? " (boundary)" : "");
    if (next.IsOnBoundary() && !next.IsOutside()) {
      std::printf("  exit climb / entry before relocation: %s, last exited %s\n",
                  next.Top() ? next.Top()->GetLabel().c_str() : "outside",
                  next.GetLastExited() ? next.GetLastExited()->GetLabel().c_str() : "none");
      vecgeom::BVHNavigator::RelocateToNextVolume(point, dir, next);
      std::printf("  relocation with the final direction: %s (%s the recorded state)\n",
                  next.Top() ? next.Top()->GetLabel().c_str() : "outside",
                  SameState(next, to) ? "same as" : "DIFFERENT from");
    }
  } else {
    std::printf("  curved step in a field: not recomputed\n");
  }
}

/// Rebuild a Geant4 history from (physical volume name, copy number) per level.
bool ReadHistory(std::istream &in, G4VPhysicalVolume *world, G4NavigationHistory &history, std::string &error)
{
  int n;
  in >> n;
  G4VPhysicalVolume *volume = world;
  for (int level = 0; level < n; ++level) {
    std::string name;
    int copy;
    in >> name >> copy;
    if (level == 0) {
      if (world->GetName() != name) error = "world name mismatch: " + world->GetName() + " != " + name;
      history.SetFirstEntry(world);
      continue;
    }
    G4VPhysicalVolume *next = nullptr;
    auto *mother            = volume->GetLogicalVolume();
    for (std::size_t d = 0; d < mother->GetNoDaughters() && !next; ++d) {
      auto *daughter = mother->GetDaughter(d);
      if (daughter->GetName() == name && daughter->GetCopyNo() == copy) next = daughter;
    }
    if (!next) {
      error = "no daughter " + name + "#" + std::to_string(copy) + " in " + mother->GetName();
      return false;
    }
    history.NewLevel(next, kNormal, copy);
    volume = next;
  }
  return error.empty();
}

std::string PathString(G4NavigationHistory const &history)
{
  std::string out;
  for (std::size_t level = 1; level <= history.GetDepth(); ++level)
    out += "/" + history.GetVolume(level)->GetName();
  return out;
}

void ReplayHandoff(std::istream &in, std::string const &id, G4VPhysicalVolume *world, G4Navigator &navigator)
{
  G4NavigationHistory exited, expected;
  std::string error;
  if (!ReadHistory(in, world, exited, error) || !ReadHistory(in, world, expected, error)) {
    std::printf("H %s: cannot rebuild the Geant4 paths (%s)\n", id.c_str(), error.c_str());
    return;
  }
  const Vec b = ReadVec(in), h = ReadVec(in), d = ReadVec(in);
  const G4ThreeVector boundary(b.x(), b.y(), b.z()), handed(h.x(), h.y(), h.z()), dir(d.x(), d.y(), d.z());
  std::printf("H %s: %s -> %s\n", id.c_str(), PathString(exited).c_str(), PathString(expected).c_str());
  G4TouchableHistory touchable(expected);
  for (auto const &[what, point] : {std::pair{"boundary point", boundary}, std::pair{"handed point", handed}}) {
    navigator.ResetHierarchyAndLocate(point, dir, touchable);
    std::unique_ptr<G4TouchableHistory> located(navigator.CreateTouchableHistory());
    std::printf("  Geant4 relocation at the %s: %s (distances in kTolerance)\n", what,
                PathString(*located->GetHistory()).c_str());
    const std::vector<G4NavigationHistory const *> histories{&exited, &expected, located->GetHistory()};
    for (auto const *history : histories) {
      const std::size_t level = history->GetDepth();
      G4AffineTransform toLocal(history->GetTransform(level));
      auto const *solid  = history->GetVolume(level)->GetLogicalVolume()->GetSolid();
      const auto local   = toLocal.TransformPoint(point);
      const auto inside  = solid->Inside(local);
      const double nDotD = solid->SurfaceNormal(local).dot(toLocal.TransformAxis(dir));
      std::printf("    %-40s G4 Inside %-7s DistanceToIn %10.3g DistanceToOut %10.3g n.d %.4g\n",
                  history->GetVolume(level)->GetName().c_str(),
                  InsideName(inside == kInside    ? vecgeom::kInside
                             : inside == kSurface ? vecgeom::kSurface
                                                  : vecgeom::kOutside),
                  inside == kOutside ? solid->DistanceToIn(local) / vecgeom::kTolerance : 0.,
                  inside != kOutside ? solid->DistanceToOut(local) / vecgeom::kTolerance : 0., nDotD);
    }
  }
}

} // namespace

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <file.gdml> [boolean optimization threshold] < records.txt\n"
              << "Records are prepared by replay_geometry_validation.py.\n";
    return 1;
  }
  if (argc > 2) vecgeom::GeoManager::Instance().SetBooleanOptimizationThreshold(std::atoi(argv[2]));
  G4GDMLParser parser;
  parser.Read(argv[1], false);
  G4VPhysicalVolume *world = parser.GetWorldVolume();
  G4GeometryManager::GetInstance()->CloseGeometry();
  AdePTGeometryBridge::CreateVecGeomWorld(world);
  vecgeom::BVHManager::Init();
  G4Navigator navigator;
  navigator.SetWorldVolume(world);

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string type, id;
    if (!(in >> type >> id)) continue;
    if (type == "C")
      ReplayCrossing(in, id);
    else if (type == "H")
      ReplayHandoff(in, id, world, navigator);
    std::fflush(stdout);
  }
  return 0;
}
