// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/g4integration/returned_steps/HandoffValidator.hh>
#include <AdePT/transport/geometry_validation/ValidationJson.hh>

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
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <tuple>

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

/// Path from the world as (physical volume name, copy number) per level, for replays.
std::vector<std::pair<std::string, int>> PathOf(G4NavigationHistory const &path)
{
  std::vector<std::pair<std::string, int>> out;
  for (std::size_t level = 0; level <= path.GetDepth(); ++level)
    out.emplace_back(Name(path, level), path.GetReplicaNo(level));
  return out;
}

std::string PathJson(std::vector<std::pair<std::string, int>> const &path)
{
  using namespace adept::validation_json;
  return Array(path, path.size(), [](std::pair<std::string, int> const &level) {
    return "[" + Str(level.first) + "," + std::to_string(level.second) + "]";
  });
}

void PrintCounts(std::ostream &out, std::string const &title, HandoffValidator::Counts const &c)
{
  out << "HandoffValidation " << title << ": " << c.exits << " tracks left the GPU regions\n"
      << "HandoffValidation   tolerance band: " << c.bandErrors << " boundary points beyond Geant4's band ("
      << c.bandExited << " inside a volume being left, " << c.bandEntered << " outside a volume being entered, "
      << c.bandOff << " not on any crossed surface)\n"
      << "HandoffValidation   normal at the exit point: " << c.leaving << " leaving through the crossed surface, "
      << c.reversed << " direction pointing back, " << c.tangent << " tangent\n"
      << "HandoffValidation   relocation: " << c.confirmed << " confirmed, " << c.reversal
      << " direction reversed at the boundary, " << c.coincident << " coincident surface, " << c.pushCrossing
      << " surface crossed by the handoff push, " << c.unexplained << " unexplained\n"
      << "HandoffValidation   proven geometry overlaps: " << c.bandOverlap << " of the band errors, "
      << c.unexplainedOverlap << " of the unexplained relocations\n";
}

} // namespace

HandoffValidator::Counts &HandoffValidator::Counts::operator+=(Counts const &o)
{
  exits += o.exits;
  confirmed += o.confirmed;
  reversal += o.reversal;
  coincident += o.coincident;
  pushCrossing += o.pushCrossing;
  bandErrors += o.bandErrors;
  bandExited += o.bandExited;
  bandEntered += o.bandEntered;
  bandOff += o.bandOff;
  leaving += o.leaving;
  reversed += o.reversed;
  tangent += o.tangent;
  unexplained += o.unexplained;
  bandOverlap += o.bandOverlap;
  unexplainedOverlap += o.unexplainedOverlap;
  return *this;
}

/// Process-wide summary of the validators of all worker threads. Each validator merges into it when its worker
/// ends; the last one to release it prints the totals and writes the file.
class HandoffValidator::Summary {
public:
  Summary(std::string prefix, unsigned int capacity) : fPrefix(std::move(prefix)), fCapacity(capacity) {}

  static std::shared_ptr<Summary> Get(std::string const &prefix, unsigned int capacity)
  {
    static std::mutex mutex;
    static std::weak_ptr<Summary> instance;
    std::lock_guard<std::mutex> lock(mutex);
    auto summary = instance.lock();
    if (!summary) instance = summary = std::make_shared<Summary>(prefix, capacity);
    return summary;
  }

  void Merge(Counts const &counts, std::map<std::string, OverlapEntry> const &overlaps, std::vector<Record> &records)
  {
    std::lock_guard<std::mutex> lock(fMutex);
    ++fWorkers;
    fCounts += counts;
    for (auto const &[pair, entry] : overlaps) {
      auto &total = fOverlaps[pair];
      total.count += entry.count;
      total.depth   = std::max(total.depth, entry.depth);
      total.witness = std::abs(entry.witness) > std::abs(total.witness) ? entry.witness : total.witness;
    }
    fRecordsSeen += records.size();
    std::move(records.begin(), records.end(), std::back_inserter(fRecords));
  }

  ~Summary()
  {
    PrintCounts(std::cout, "total (" + std::to_string(fWorkers) + " worker threads)", fCounts);
    for (auto const &[pair, entry] : fOverlaps)
      std::cout << "HandoffValidation   proven " << pair << " (" << entry.count << " tracks, evidence up to "
                << entry.depth << " mm, witness up to " << entry.witness << " mm along the direction)\n";
    std::cout << "HandoffValidation total: " << (fCounts.Passed() ? "PASSED" : "FAILED") << std::endl;
    if (fPrefix.empty()) return;
    // Records arrive per worker thread: sort them for reproducible files.
    std::sort(fRecords.begin(), fRecords.end(),
              [](Record const &a, Record const &b) { return std::tie(a.event, a.track) < std::tie(b.event, b.track); });
    if (fRecords.size() > fCapacity) fRecords.resize(fCapacity);
    Write(fPrefix + "_handoff.json");
  }

private:
  void Write(std::string const &fileName) const
  {
    using namespace adept::validation_json;
    std::ofstream json(fileName);
    if (!json) {
      std::cout << "HandoffValidation   cannot write " << fileName << std::endl;
      return;
    }
    auto const &c = fCounts;
    json << "{\n\"format\": \"adept-geometry-validation-handoff-1\",\n\"worker_threads\": " << fWorkers
         << ",\n\"exits\": " << c.exits << ",\n\"tolerance_band\": {\"errors\": " << c.bandErrors
         << ", \"inside_left\": " << c.bandExited << ", \"outside_entered\": " << c.bandEntered
         << ", \"off_surface\": " << c.bandOff << ", \"proven_overlap\": " << c.bandOverlap << "}"
         << ",\n\"normal_at_exit\": {\"leaving\": " << c.leaving << ", \"pointing_back\": " << c.reversed
         << ", \"tangent\": " << c.tangent << "}" << ",\n\"relocation\": {\"confirmed\": " << c.confirmed
         << ", \"direction_reversed\": " << c.reversal << ", \"coincident_surface\": " << c.coincident
         << ", \"crossed_by_handoff_push\": " << c.pushCrossing << ", \"unexplained\": " << c.unexplained
         << ", \"unexplained_proven_overlap\": " << c.unexplainedOverlap
         << "},\n\"passed\": " << (c.Passed() ? "true" : "false") << ",\n\"proven_overlaps\": [";
    bool first = true;
    for (auto const &[pair, entry] : fOverlaps) {
      json << (first ? "\n  " : ",\n  ") << "{\"pair\": " << Str(pair) << ", \"tracks\": " << entry.count
           << ", \"depth_mm\": " << Num(entry.depth) << ", \"witness_mm\": " << Num(entry.witness) << "}";
      first = false;
    }
    json << "],\n\"records_seen\": " << fRecordsSeen << ",\n\"records\": [";
    for (std::size_t i = 0; i < fRecords.size(); ++i) {
      auto const &r = fRecords[i];
      double b[3]   = {r.boundary.x(), r.boundary.y(), r.boundary.z()};
      double h[3]   = {r.handed.x(), r.handed.y(), r.handed.z()};
      double d[3]   = {r.direction.x(), r.direction.y(), r.direction.z()};
      json << (i ? ",\n  " : "\n  ") << "{\"category\": " << Str(r.category)
           << ", \"band_error\": " << (r.band.empty() ? std::string("null") : Str(r.band))
           << ", \"proven_overlap\": " << (r.provenOverlap ? "true" : "false") << ", \"event\": " << r.event
           << ", \"track\": " << r.track << ", \"boundary_point\": " << Array3(b) << ", \"handed_point\": " << Array3(h)
           << ", \"direction\": " << Array3(d) << ", \"outward_n_dot_d\": " << Num(r.outwardNDotD)
           << ", \"exited\": " << PathJson(r.exited) << ", \"expected\": " << PathJson(r.expected)
           << ", \"located\": " << PathJson(r.located) << "}";
    }
    json << "\n]\n}\n";
    std::cout << "HandoffValidation   written " << fileName << " (" << fRecords.size() << " replayable records)"
              << std::endl;
  }

  std::mutex fMutex;
  std::string fPrefix;
  unsigned int fCapacity{0};
  int fWorkers{0};
  Counts fCounts;
  std::map<std::string, OverlapEntry> fOverlaps;
  std::vector<Record> fRecords;
  std::size_t fRecordsSeen{0};
};

HandoffValidator::HandoffValidator(std::string const &outputPrefix, unsigned int recordCapacity)
    : fNavigator(std::make_unique<G4Navigator>()), fRecordCapacity(recordCapacity),
      fSummary(Summary::Get(outputPrefix, recordCapacity))
{
  fNavigator->SetWorldVolume(
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());
}

HandoffValidator::~HandoffValidator()
{
  fSummary->Merge(fCounts, fOverlaps, fRecords);
}

void HandoffValidator::AddRecord(Record &&record)
{
  if (fRecords.size() < fRecordCapacity) fRecords.push_back(std::move(record));
}

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
  for (double witness : {0., -2., 2., -10., 10., -100., 100.})
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
    entry.witness = std::abs(witness) > std::abs(entry.witness) ? witness : entry.witness;
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
                             G4ThreeVector const &direction, int event, unsigned long long track)
{
  auto const &from = *exited->GetHistory();
  auto const &to   = *expected->GetHistory();
  ++fCounts.exits;

  // Levels below the deepest common volume are crossed: those of `from` are left, those of `to` are entered.
  std::size_t common = 0;
  while (common < from.GetDepth() && common < to.GetDepth() && SameLevel(from, to, common + 1))
    ++common;

  // 1. Tolerance band, with Geant4's own classification (band +-kCarTolerance/2) at the step end point: it must not
  //    be strictly inside a volume being left, nor strictly outside a volume being entered, and it must lie on the
  //    surface of at least one crossed volume (the surface the GPU transport stopped on).
  bool bandError = false;
  std::string band;
  // Crossed surface: the deepest left volume, else the outermost entered volume, whose surface holds the point.
  // `outwardNd` is the normal of that surface . direction, oriented out of the side being left.
  bool onCrossedSurface = false;
  double outwardNd      = 0.;
  for (std::size_t level = from.GetDepth(); level > common; --level) {
    const auto view = View(from, level, boundaryPoint, direction);
    if (view.inside == kInside && !bandError) {
      bandError = true;
      band      = "inside the volume being left: " + Name(from, level);
      ++fCounts.bandExited;
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
      band      = "outside the volume being entered: " + Name(to, level);
      ++fCounts.bandEntered;
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
    band      = "not on the surface of any crossed volume";
    ++fCounts.bandOff;
    Detail("boundary point not on the surface of any crossed volume", boundaryPoint, direction,
           "left " + Name(from, from.GetDepth()) + ", reached " + Name(to, to.GetDepth()));
  }
  bool provenOverlap = false;
  if (bandError) {
    ++fCounts.bandErrors;
    provenOverlap = ProveOverlap(from, to, boundaryPoint, direction);
    if (provenOverlap) ++fCounts.bandOverlap;
  }

  // 2. Normal at the exit point: does the final direction leave through the crossed surface?
  if (onCrossedSurface) {
    if (outwardNd > 0.)
      ++fCounts.leaving;
    else if (outwardNd < 0.)
      ++fCounts.reversed;
    else
      ++fCounts.tangent;
  }

  // 3. Relocation, as G4SteppingManager::SetInitialStep does for a track carrying a touchable.
  fNavigator->ResetHierarchyAndLocate(handedPoint, direction, *static_cast<G4TouchableHistory *>(expected()));
  std::unique_ptr<G4TouchableHistory> locatedTouchable(fNavigator->CreateTouchableHistory());
  auto const &located = *locatedTouchable->GetHistory();
  std::string outcome = Classify(from, to, located, common, onCrossedSurface, outwardNd, boundaryPoint, handedPoint,
                                 direction, expected, provenOverlap);
  if (outcome == "confirmed" && !bandError) return;
  Record record;
  record.category      = outcome;
  record.band          = band;
  record.provenOverlap = provenOverlap;
  record.event         = event;
  record.track         = track;
  record.boundary      = boundaryPoint;
  record.handed        = handedPoint;
  record.direction     = direction;
  record.outwardNDotD  = outwardNd;
  record.exited        = PathOf(from);
  record.expected      = PathOf(to);
  record.located       = PathOf(located);
  AddRecord(std::move(record));
}

std::string HandoffValidator::Classify(G4NavigationHistory const &from, G4NavigationHistory const &to,
                                       G4NavigationHistory const &located, std::size_t common, bool onCrossedSurface,
                                       double outwardNd, G4ThreeVector const &boundaryPoint,
                                       G4ThreeVector const &handedPoint, G4ThreeVector const &direction,
                                       G4TouchableHandle const &expected, bool &provenOverlap)
{
  if (SamePath(located, to)) {
    ++fCounts.confirmed;
    return "confirmed";
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
      ++fCounts.reversal;
      return "direction reversed";
    }
  }

  // Coincident surface: the located path enters a further volume whose surface also holds the boundary point, with
  // the direction going into it (a zero step for the GPU navigation).
  if (Contains(located, to, to.GetDepth()) && located.GetDepth() > to.GetDepth()) {
    const auto first = View(located, to.GetDepth() + 1, boundaryPoint, direction);
    if (first.inside == kSurface && first.nDotD < 0.) {
      ++fCounts.coincident;
      return "coincident surface";
    }
  }

  // Boundary crossed by the handoff push: at the GPU end point itself Geant4 confirms the GPU path, the push along the
  // direction (applied before the track is given to Geant4) moved it across a further surface.
  if (handedPoint != boundaryPoint) {
    fNavigator->ResetHierarchyAndLocate(boundaryPoint, direction, *static_cast<G4TouchableHistory *>(expected()));
    std::unique_ptr<G4TouchableHistory> rawTouchable(fNavigator->CreateTouchableHistory());
    if (SamePath(*rawTouchable->GetHistory(), to)) {
      ++fCounts.pushCrossing;
      return "crossed by handoff push";
    }
  }

  ++fCounts.unexplained;
  if (ProveOverlap(located, to, handedPoint, direction) || ProveOverlap(from, to, boundaryPoint, direction)) {
    ++fCounts.unexplainedOverlap;
    provenOverlap = true;
    return "unexplained";
  }
  std::ostringstream volumes;
  volumes << "left " << Name(from, from.GetDepth()) << ", reached " << Name(to, to.GetDepth()) << " (depth "
          << to.GetDepth() << "), Geant4 locates " << Name(located, located.GetDepth()) << " (depth "
          << located.GetDepth() << "), normal . direction at the exit point " << outwardNd;
  Detail("relocation not explained by the geometry (overlap or solid disagreement?)", boundaryPoint, direction,
         volumes.str());
  return "unexplained";
}

void HandoffValidator::Report(std::ostream &out, int threadId) const
{
  PrintCounts(out, "thread " + std::to_string(threadId), fCounts);
  for (auto const &[pair, entry] : fOverlaps)
    out << "HandoffValidation   proven " << pair << " (" << entry.count << " tracks, evidence up to " << entry.depth
        << " mm, witness up to " << entry.witness << " mm along the direction)\n";
  out << "HandoffValidation: " << (Passed() ? "PASSED" : "FAILED") << std::endl;
}
