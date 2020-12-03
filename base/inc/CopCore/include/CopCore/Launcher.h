// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Launcher.h
 * @brief Backend-dependent abstraction for parallel function execution
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef COPCORE_LAUNCHER_H_
#define COPCORE_LAUNCHER_H_

#include <CopCore/Global.h>

namespace copcore {

#ifdef TARGET_DEVICE_CUDA
namespace kernel_launcher_impl {

template <class Function, class... Args>
__global__ void kernel_dispatch(int data_size, Function device_func_select, const Args... args)
{
  // Initiate a grid-size loop to maximize reuse of threads and CPU compatibility, keeping adressing within warps
  // unit-stride
  for (auto id = blockIdx.x * blockDim.x + threadIdx.x; id < data_size; id += blockDim.x * gridDim.x) {

    device_func_select(id, args...);
  }
}
} // End namespace kernel_launcher_impl
#endif

template <BackendType backend>
class LauncherBase {
public:
  using LaunchGrid_t = launch_grid<backend>;
  using Stream_t     = typename copcore::StreamType<backend>::value_type;

protected:
  int fDeviceId{0}; ///< device id (GPU for CUDA, socket for CPU)
  // LaunchGrid_t fGrid{{0}, {0}}; ///< launch grid to be used if set by user
  Stream_t fStream{0}; ///< stream id for CUDA, not used for CPU)

public:
  LauncherBase(Stream_t stream) : fStream{stream} {}

  // void SetDevice(int device) { fDeviceId = device; }
  int GetDevice() const { return fDeviceId; }

  void SetStream(Stream_t stream) { fStream = stream; }
  Stream_t GetStream() const { return fStream; }
}; // end class LauncherBase

/** @brief Launcher for generic backend */
template <BackendType backend>
class Launcher : protected LauncherBase<backend> {
public:
  using LaunchGrid_t = launch_grid<backend>;
  /** @brief Generic backend launch method. Implementation done in specializations */
  template <class FunctionPtr, class... Args>
  int Run(FunctionPtr, int, LaunchGrid_t, const Args &...) const
  {
    // Not implemented backend launches will end-up here
    std::string backend_name(copcore::BackendName(backend));
    COPCORE_EXCEPTION("Launcher::Launch: No implementation available for " + backend_name);
    return 1;
  }
};

#ifdef TARGET_DEVICE_CUDA
/** @brief Specialization of Launcher for the CUDA backend */
template <>
class Launcher<BackendType::CUDA> : public LauncherBase<BackendType::CUDA> {
private:
  int fNumSMs;     ///< number of streaming multi-processors

public:
  Launcher(Stream_t stream = 0) : LauncherBase(stream)
  {
    cudaGetDevice(&fDeviceId);
    cudaDeviceGetAttribute(&fNumSMs, cudaDevAttrMultiProcessorCount, fDeviceId);
  }

  template <class DeviceFunctionPtr, class... Args>
  int Run(DeviceFunctionPtr func, int n_elements, LaunchGrid_t grid, const Args &... args) const
  {
    constexpr unsigned int warpsPerSM = 32; // we should target a reasonable occupancy
    constexpr unsigned int block_size = 256;

    // Compute the launch grid.
    if (!n_elements) return 0;

    // Adjust automatically the execution grid. Optimal occupancy:
    // nMaxThreads = fNumSMs * warpsPerSM * 32; grid_size = nmaxthreads / block_size
    // if n_elements < nMaxThreads we reduce the grid size to minimize null threads
    LaunchGrid_t exec_grid{grid};
    if (grid[1].x == 0) {
      unsigned int grid_size =
          std::min(warpsPerSM * fNumSMs * 32 / block_size, (n_elements + block_size - 1) / block_size);
      exec_grid[0].x = grid_size;
      exec_grid[1].x = block_size;
      // std::cout << "grid_size = " << grid_size << "  block_size = " << block_size << std::endl;
    }

    // launch the kernel
    kernel_launcher_impl::kernel_dispatch<<<exec_grid[0], exec_grid[1], 0, fStream>>>(n_elements, func, args...);
    COPCORE_CUDA_CHECK(cudaGetLastError());
    return 0;
  }

  void WaitStream() const
  {
    if (fStream)
      COPCORE_CUDA_CHECK(cudaStreamSynchronize(fStream));
    else
      COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  }

  static void WaitDevice() { COPCORE_CUDA_CHECK(cudaDeviceSynchronize()); }

}; // End  class Launcher<BackendType::CUDA>
#endif

/** @brief Specialization of Launcher for the CPU backend */
template <>
class Launcher<BackendType::CPU> : public LauncherBase<BackendType::CPU> {
public:
  Launcher(Stream_t stream = 0) : LauncherBase(stream) {}

  template <class HostFunctionPtr, class... Args>
  int Run(HostFunctionPtr func, int n_elements, LaunchGrid_t /*grid*/, const Args &... args) const
  {
#pragma omp parallel for
    for (int i = 0; i < n_elements; ++i) {
      func(i, args...);
    }
    return 0;
  }
  void WaitStream() const {}
  static void WaitDevice() {}

}; // End class Launcher<BackendType::CPU>

} // End namespace copcore

#endif // COPCORE_LAUNCHER_H_
