// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/returned_steps/HandoffValidator.hh>

#include <G4AffineTransform.hh>
#include <G4GeometryTolerance.hh>
#include <G4LogicalVolume.hh>
#include <G4NavigationHistory.hh>
#include <G4Navigator.hh>
#include <G4StateManager.hh>
#include <G4TouchableHistory.hh>
#include <G4TransportationManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSolid.hh>
#include <G4ios.hh>

#include <cstdio>
#include <iterator>
#include <numeric>
#include <string>

namespace {

constexpr int kMaxPrinted = 10; // unexpected handoffs printed in detail, per thread and run

bool SameLevel(G4NavigationHistory const &a, G4NavigationHistory const &b, std::size_t level)
{
  return a.GetVolume(level) == b.GetVolume(level) && a.GetReplicaNo(level) == b.GetReplicaNo(level);
}

/// True if `path` goes through `reference` down to `level`.
bool Contains(G4NavigationHistory const &path, G4NavigationHistory const &reference, std::size_t level)
{
  if (path.GetDepth() < level) return false;
  for (std::size_t i = 0; i <= level; ++i)
    if (!SameLevel(path, reference, i)) return false;
  return true;
}

bool SamePath(G4NavigationHistory const &a, G4NavigationHistory const &b)
{
  return a.GetDepth() == b.GetDepth() && Contains(a, b, b.GetDepth());
}

/// Deepest level shared by both paths.
std::size_t CommonLevel(G4NavigationHistory const &a, G4NavigationHistory const &b)
{
  std::size_t common = 0;
  while (common < a.GetDepth() && common < b.GetDepth() && SameLevel(a, b, common + 1))
    ++common;
  return common;
}

/// A parameterised or replicated copy shares its solid with the other copies, so the solid of the logical volume
/// does not have this copy's dimensions.
bool HasReplicas(G4NavigationHistory const &path)
{
  for (std::size_t level = 0; level <= path.GetDepth(); ++level)
    if (path.GetVolumeType(level) != kNormal) return true;
  return false;
}

std::string Name(G4NavigationHistory const *path)
{
  if (path == nullptr || path->GetTopVolume() == nullptr) return "outside the world";
  return std::string(path->GetTopVolume()->GetName()) + "#" + std::to_string(path->GetTopReplicaNo());
}

/// The point and direction in the frame of the solid of `level`, and Geant4's Inside there.
struct LevelView {
  G4VSolid const *solid;
  G4ThreeVector point, direction;
  EInside inside;
  /// Outward normal . direction, on the surface.
  double NDotD() const { return inside == kSurface ? solid->SurfaceNormal(point).dot(direction) : 0.; }
  /// > 0 strictly inside, < 0 strictly outside, 0 on the surface.
  double StrictDistance() const
  {
    return inside == kInside ? solid->DistanceToOut(point) : inside == kOutside ? -solid->DistanceToIn(point) : 0.;
  }
};

LevelView View(G4NavigationHistory const &path, std::size_t level, G4ThreeVector const &point,
               G4ThreeVector const &direction = {})
{
  const G4AffineTransform toLocal = path.GetTransform(level);
  auto const *solid               = path.GetVolume(level)->GetLogicalVolume()->GetSolid();
  const G4ThreeVector local       = toLocal.TransformPoint(point);
  return {solid, local, toLocal.TransformAxis(direction), solid->Inside(local)};
}

/// The point is strictly inside volumes of both paths in different branches (overlap), or strictly inside a volume
/// and strictly outside one of its ancestors (extrusion).
bool ProveOverlapAt(G4NavigationHistory const &a, G4NavigationHistory const &b, G4ThreeVector const &point)
{
  const std::size_t common = CommonLevel(a, b);
  auto deepestInside       = [&](G4NavigationHistory const &path) {
    std::size_t inside = 0;
    for (std::size_t level = common + 1; level <= path.GetDepth(); ++level)
      if (View(path, level, point).StrictDistance() > 0.) inside = level;
    return inside;
  };
  const std::size_t insideA = deepestInside(a), insideB = deepestInside(b);
  if (insideA > 0 && insideB > 0) return true;
  for (auto [path, inside] : {std::pair{&a, insideA}, std::pair{&b, insideB}})
    for (std::size_t level = 1; level < inside; ++level)
      if (View(*path, level, point).StrictDistance() < 0.) return true;
  return false;
}

/// Overlap or extrusion at the point, or within 100 tolerances of it along the direction.
bool ProveOverlap(G4NavigationHistory const &a, G4NavigationHistory const &b, G4ThreeVector const &point,
                  G4ThreeVector const &direction)
{
  const double tolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
  for (double witness : {0., -2., 2., -10., 10., -100., 100.})
    if (ProveOverlapAt(a, b, point + witness * tolerance * direction)) return true;
  return false;
}

} // namespace

char const *HandoffOutcomeName(HandoffOutcome outcome)
{
  static constexpr char const *names[] = {
      "confirmed", "coincident surface", "crossed by the handoff push", "overlap",
      "replicas",  "band error",         "direction reversed",          "other path"};
  static_assert(std::size(names) == std::size_t(HandoffOutcome::Count));
  return names[std::size_t(outcome)];
}

HandoffCounts &HandoffCounts::operator+=(HandoffCounts const &other)
{
  for (std::size_t i = 0; i < n.size(); ++i)
    n[i] += other.n[i];
  return *this;
}

std::uint64_t HandoffCounts::Total() const
{
  return std::accumulate(n.begin(), n.end(), std::uint64_t{0});
}

std::uint64_t HandoffCounts::Inconclusive() const
{
  return (*this)[HandoffOutcome::PushCrossing] + (*this)[HandoffOutcome::Overlap] + (*this)[HandoffOutcome::Replicas];
}

std::uint64_t HandoffCounts::Unexpected() const
{
  return (*this)[HandoffOutcome::BandError] + (*this)[HandoffOutcome::DirectionReversed] +
         (*this)[HandoffOutcome::OtherPath];
}

HandoffCounts::Verdict HandoffCounts::GetVerdict() const
{
  if (Unexpected() > 0) return Verdict::Failed;
  return Conclusive() > 0 ? Verdict::Passed : Verdict::NoConclusiveHandoff;
}

void HandoffCounts::Print(std::ostream &out) const
{
  auto list = [&](std::initializer_list<HandoffOutcome> outcomes) {
    std::string text;
    for (auto outcome : outcomes)
      text += (text.empty() ? "" : ", ") + std::to_string((*this)[outcome]) + " " + HandoffOutcomeName(outcome);
    return text;
  };
  const double inconclusivePercent = Total() > 0 ? 100. * Inconclusive() / Total() : 0.;
  out << "HandoffValidation: " << Total()
      << " tracks left the GPU regions: " << list({HandoffOutcome::Confirmed, HandoffOutcome::CoincidentSurface})
      << "; inconclusive " << Inconclusive() << " (" << inconclusivePercent
      << " %): " << list({HandoffOutcome::PushCrossing, HandoffOutcome::Overlap, HandoffOutcome::Replicas})
      << "; unexpected " << Unexpected() << ": "
      << list({HandoffOutcome::BandError, HandoffOutcome::DirectionReversed, HandoffOutcome::OtherPath}) << std::endl;
  const Verdict verdict = GetVerdict();
  out << "HandoffValidation: "
      << (verdict == Verdict::Failed   ? "FAILED"
          : verdict == Verdict::Passed ? "PASSED"
                                       : "NO CONCLUSIVE HANDOFF")
      << " (" << Conclusive() << " of " << Total() << " handoffs checked, " << Unexpected() << " unexpected)"
      << std::endl;
}

HandoffValidator::HandoffValidator() : fNavigator(std::make_unique<G4Navigator>())
{
  fNavigator->SetWorldVolume(
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());
}

HandoffValidator::~HandoffValidator() = default;

HandoffOutcome HandoffValidator::Check(G4TouchableHandle const &exited, G4TouchableHandle const &expected,
                                       G4ThreeVector const &endPoint, G4ThreeVector const &handedPoint,
                                       G4ThreeVector const &direction)
{
  auto const &from = *exited->GetHistory();
  auto const &to   = *expected->GetHistory();
  // Checked before any solid query: the solids of replicas do not have the dimensions of this copy.
  const HandoffOutcome outcome = HasReplicas(from) || HasReplicas(to)
                                     ? HandoffOutcome::Replicas
                                     : Classify(from, to, endPoint, handedPoint, direction, expected);
  ++fCounts[outcome];
  return outcome;
}

HandoffOutcome HandoffValidator::Classify(G4NavigationHistory const &from, G4NavigationHistory const &to,
                                          G4ThreeVector const &endPoint, G4ThreeVector const &handedPoint,
                                          G4ThreeVector const &direction, G4TouchableHandle const &expected)
{
  // 1. Tolerance band, with Geant4's own Inside: the end point must not be strictly inside a volume being left (the
  //    levels of `from` below the common volume), nor strictly outside a volume being entered (those of `to`), and
  //    it must be on the surface of one of them. `outwardNd` is the normal of the first such surface . direction,
  //    oriented out of the side being left.
  const std::size_t common = CommonLevel(from, to);
  bool bandError           = false;
  bool onCrossedSurface    = false;
  double outwardNd         = 0.;
  for (std::size_t level = from.GetDepth(); level > common; --level) {
    const auto view = View(from, level, endPoint, direction);
    bandError |= view.inside == kInside;
    if (view.inside == kSurface && !onCrossedSurface) onCrossedSurface = true, outwardNd = view.NDotD();
  }
  for (std::size_t level = common + 1; level <= to.GetDepth(); ++level) {
    const auto view = View(to, level, endPoint, direction);
    bandError |= view.inside == kOutside;
    if (view.inside == kSurface && !onCrossedSurface) onCrossedSurface = true, outwardNd = -view.NDotD();
  }
  bandError |= !onCrossedSurface;

  // 2. Relocation, as G4SteppingManager::SetInitialStep does for a track that carries a touchable (outside the
  //    world, Geant4 kills the track).
  const auto locatedTouchable = Locate(handedPoint, direction, expected);
  auto const *located         = locatedTouchable ? locatedTouchable->GetHistory() : nullptr;
  if (located && HasReplicas(*located)) return HandoffOutcome::Replicas;

  HandoffOutcome outcome = HandoffOutcome::OtherPath;
  if (bandError) {
    outcome = HandoffOutcome::BandError;
  } else if (located && SamePath(*located, to)) {
    return HandoffOutcome::Confirmed;
  } else if (outwardNd < 0.) {
    outcome = HandoffOutcome::DirectionReversed;
  } else {
    // Geant4 at the end point itself, which the handoff push does not reach.
    const auto atEndTouchable = Locate(endPoint, direction, expected);
    auto const *atEnd         = atEndTouchable ? atEndTouchable->GetHistory() : nullptr;
    if (atEnd && HasReplicas(*atEnd)) return HandoffOutcome::Replicas;
    // Geant4 goes deeper, into a volume that starts on the crossed surface and that the direction enters (the
    // volumes below it are inside it).
    auto coincident = [&](G4NavigationHistory const *path) {
      if (!path || path->GetDepth() <= to.GetDepth() || !Contains(*path, to, to.GetDepth())) return false;
      const auto view = View(*path, to.GetDepth() + 1, endPoint, direction);
      return view.inside == kSurface && view.NDotD() < 0.;
    };
    // Accepted only if the push changes nothing: it could skip a thin volume at the coincident surface.
    if (coincident(located) && atEnd && SamePath(*atEnd, *located)) return HandoffOutcome::CoincidentSurface;
    if (atEnd && (SamePath(*atEnd, to) || coincident(atEnd))) return HandoffOutcome::PushCrossing;
  }

  if (ProveOverlap(from, to, endPoint, direction) || (located && ProveOverlap(*located, to, handedPoint, direction)))
    return HandoffOutcome::Overlap;
  PrintUnexpected(outcome, from, to, located, endPoint, direction);
  return outcome;
}

std::unique_ptr<G4TouchableHistory> HandoffValidator::Locate(G4ThreeVector const &point, G4ThreeVector const &direction,
                                                             G4TouchableHandle const &start)
{
  if (!fNavigator->ResetHierarchyAndLocate(point, direction, *static_cast<G4TouchableHistory *>(start())))
    return nullptr;
  return std::unique_ptr<G4TouchableHistory>(fNavigator->CreateTouchableHistory());
}

void HandoffValidator::PrintUnexpected(HandoffOutcome outcome, G4NavigationHistory const &from,
                                       G4NavigationHistory const &to, G4NavigationHistory const *located,
                                       G4ThreeVector const &endPoint, G4ThreeVector const &direction)
{
  if (fPrinted++ >= kMaxPrinted) return;
  std::printf("HandoffValidation unexpected (%s): left %s, reached %s, Geant4 locates %s; end point (%.17g, %.17g, "
              "%.17g) direction (%.17g, %.17g, %.17g)\n",
              HandoffOutcomeName(outcome), Name(&from).c_str(), Name(&to).c_str(), Name(located).c_str(), endPoint.x(),
              endPoint.y(), endPoint.z(), direction.x(), direction.y(), direction.z());
}

void HandoffValidator::EndOfRun(std::ostream &out)
{
  fCounts.Print(out);
  fCounts  = {};
  fPrinted = 0;
}

G4bool HandoffValidator::Notify(G4ApplicationState requestedState)
{
  if (G4StateManager::GetStateManager()->GetCurrentState() == G4State_GeomClosed && requestedState == G4State_Idle)
    EndOfRun(G4cout);
  return true;
}
