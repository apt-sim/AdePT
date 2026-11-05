// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/transport/kernels/WoodcockHelper.cuh>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/management/GeoManager.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>

// Geometry construction is compiled as host C++: VecGeom's default namespace
// in a CUDA translation unit denotes the device volume classes.
void BuildWoodcockTestGeometry(std::array<std::uint64_t, 7> &, std::array<int, 4> &);

namespace {
template <class T>
struct ManagedArray {
  T *data = nullptr;
  explicit ManagedArray(std::size_t count)
  {
    const auto status = cudaMallocManaged(&data, count * sizeof(T));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  ~ManagedArray() { cudaFree(data); }
  ManagedArray(const ManagedArray &)            = delete;
  ManagedArray &operator=(const ManagedArray &) = delete;
  T &operator[](std::size_t index) { return data[index]; }
};

struct GeometryCleanup {
  ~GeometryCleanup()
  {
    vecgeom::cxx::CudaManager::Instance().CleanGpu();
    vecgeom::GeoManager::Instance().Clear();
  }
};

__global__ void CheckWdtDispatch(const vecgeom::NavigationState *states, const double *energies, int count, int *roots,
                                 int *regionIndices, bool *dispatch)
{
  const int index = threadIdx.x;
  if (index >= count) return;
  const int wdtIdx     = adept::transport::GetWDTRegionIndex(states[index]);
  roots[index]         = adept::transport::FindWDTRoot(states[index], adept::transport::gWDTData, wdtIdx);
  regionIndices[index] = wdtIdx;
  dispatch[index]      = adept::transport::ShouldUseWDT(states[index], energies[index]);
}
} // namespace

TEST(WoodcockHelper, SeparatesCandidateAdmissionFromEnvelopeContainment)
{
  GeometryCleanup cleanup;
  vecgeom::GeoManager::Instance().Clear();
  std::array<std::uint64_t, 7> indices{};
  std::array<int, 4> logicalIds{};
  BuildWoodcockTestGeometry(indices, logicalIds);
  auto state = [](std::uint64_t index) {
    vecgeom::NavigationState result;
    result.SetNavIndex(index);
    return result;
  };
  const auto aLeft      = state(indices[0]);
  const auto aRight     = state(indices[1]);
  const auto bLeft      = state(indices[2]);
  const auto cLeft      = state(indices[3]);
  const auto bRight     = state(indices[4]);
  const auto bOutside   = state(indices[5]);
  const auto worldState = state(indices[6]);

  ManagedArray<adeptint::WDTRoot> roots(2);
  roots[0] = {aLeft, 7};
  roots[1] = {aRight, 9};
  ManagedArray<adeptint::WDTRegion> regions(3);
  ManagedArray<int> regionToWdt(8);
  for (int region = 0; region < 8; ++region)
    regionToWdt[region] = -1;
  for (int region = 0; region < 3; ++region) {
    regions[region]     = {0, 2, 0.25f};
    regionToWdt[region] = region;
  }
  // G4 region instance IDs can be sparse and exceed the WDT bucket count.
  regionToWdt[7] = 1;
  adeptint::WDTDeviceView view{roots.data, regions.data, regionToWdt.data, 2, 3, 10};
  // Region admission is deliberately cheap. A B track outside A may enter the
  // WDT queue, but the kernel must find no enclosing root and return it unchanged.
  // The integration regression compares that reject pass with direct normal dispatch.
  auto findRoot = [&](const vecgeom::NavigationState &navState, int regionId) {
    return adept::transport::FindWDTRoot(navState, view, adept::transport::GetWDTRegionIndex(view, regionId));
  };
  EXPECT_EQ(findRoot(aLeft, 0), 0);
  EXPECT_EQ(findRoot(bLeft, 7), 0);
  EXPECT_EQ(findRoot(cLeft, 2), 0);
  EXPECT_EQ(findRoot(bRight, 7), 1);
  EXPECT_EQ(findRoot(bOutside, 7), -1);
  EXPECT_EQ(findRoot(bLeft, -1), -1);
  EXPECT_EQ(findRoot(worldState, 3), -1);
  EXPECT_EQ(findRoot({}, 7), -1);
  EXPECT_EQ(adept::transport::GetWDTRegionIndex({}, 0), -1);

  ManagedArray<adeptint::VolAuxData> aux(vecgeom::GeoManager::Instance().GetRegisteredVolumesCount());
  for (std::size_t i = 0; i < vecgeom::GeoManager::Instance().GetRegisteredVolumesCount(); ++i)
    aux[i] = adeptint::VolAuxData{};
  aux[logicalIds[1]].fGPUregionId = 0;
  aux[logicalIds[2]].fGPUregionId = 7;
  aux[logicalIds[3]].fGPUregionId = 2;
  aux[logicalIds[0]].fGPUregionId = 3;
  ASSERT_TRUE(vecgeom::cxx::CudaManager::Instance().SynchronizeNavigationTable());
  ASSERT_EQ(cudaMemcpyToSymbol(adept::transport::gWDTData, &view, sizeof(view)), cudaSuccess);
  ASSERT_EQ(cudaMemcpyToSymbol(adept::transport::gVolAuxData, &aux.data, sizeof(aux.data)), cudaSuccess);

  constexpr int count = 9;
  ManagedArray<vecgeom::NavigationState> states(count);
  ManagedArray<int> regionIndices(count), selectedRoots(count);
  ManagedArray<double> energies(count);
  ManagedArray<bool> dispatch(count);
  const std::array<vecgeom::NavigationState, count> inputStates{aLeft,      bLeft, cLeft, bRight, bOutside,
                                                                worldState, {},    bLeft, bLeft};
  const std::array<int, count> expectedRegions{0, 1, 2, 1, 1, -1, -1, 1, 1};
  const std::array<int, count> expectedRoots{0, 0, 0, 1, -1, -1, -1, 0, 0};
  const std::array<bool, count> expectedDispatch{true, true, true, true, true, false, false, false, false};
  for (int i = 0; i < count; ++i) {
    states[i]   = inputStates[i];
    energies[i] = i == 7 ? 0.125 : (i == 8 ? 0.25 : 1.);
  }
  CheckWdtDispatch<<<1, 32>>>(states.data, energies.data, count, selectedRoots.data, regionIndices.data, dispatch.data);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(selectedRoots[i], expectedRoots[i]) << "case " << i;
    EXPECT_EQ(regionIndices[i], expectedRegions[i]) << "case " << i;
    EXPECT_EQ(dispatch[i], expectedDispatch[i]) << "case " << i;
  }

  // Empty WDT data must reject admission without dereferencing the null map.
  const adeptint::WDTDeviceView emptyView{};
  ASSERT_EQ(cudaMemcpyToSymbol(adept::transport::gWDTData, &emptyView, sizeof(emptyView)), cudaSuccess);
  CheckWdtDispatch<<<1, 32>>>(states.data, energies.data, count, selectedRoots.data, regionIndices.data, dispatch.data);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(selectedRoots[i], -1) << "case " << i;
    EXPECT_EQ(regionIndices[i], -1) << "case " << i;
    EXPECT_FALSE(dispatch[i]) << "case " << i;
  }
}
