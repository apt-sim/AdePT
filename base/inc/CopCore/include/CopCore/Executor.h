// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Executor.h
 * @brief Backend-dependent abstraction for parallel function execution
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef COPCORE_EXECUTOR_H_
#define COPCORE_EXECUTOR_H_

#include <CopCore/Global.h>

namespace copcore {

#ifdef VECCORE_CUDA
namespace kernel_executor_impl {

template <class Function, class... Args>
__global__ void kernel_dispatch(int *data_size, Function device_func_select, const Args... args)
{
  // Initiate a grid-size loop to maximize reuse of threads and CPU compatibility, keeping adressing within warps
  // unit-stride
  for (auto id = blockIdx.x * blockDim.x + threadIdx.x; id < *data_size; id += blockDim.x * gridDim.x) {

    device_func_select(id, args...);
  }
}
} // End namespace kernel_executor_impl
#endif

template <BackendType backend>
class ExecutorBase {
public:
  using LaunchGrid_t = launch_grid<backend>;
  using Stream_t     = typename copcore::StreamType<backend>::value_type;

protected:
  int fDeviceId{0}; ///< device id (GPU for CUDA, socket for CPU)
  // LaunchGrid_t fGrid{{0}, {0}}; ///< launch grid to be used if set by user
  Stream_t fStream{0}; ///< stream id for CUDA, not used for CPU)

public:
  ExecutorBase(Stream_t stream) : fStream{stream} {}

  // void SetDevice(int device) { fDeviceId = device; }
  int GetDevice() const { return fDeviceId; }

  void SetStream(Stream_t stream) { fStream = stream; }
  Stream_t GetStream() const { return fStream; }
}; // end class ExecutorBase

/** @brief Executor for generic backend */
template <BackendType backend>
class Executor : protected ExecutorBase<backend> {
public:
  using LaunchGrid_t = launch_grid<backend>;
  /** @brief Generic backend launch method. Implementation done in specializations */
  template <class FunctionPtr, class... Args>
  int Launch(FunctionPtr, int, LaunchGrid_t, const Args &...) const
  {
    // Not implemented backend launches will end-up here
    std::string backend_name(copcore::BackendName<backend>);
    COPCORE_EXCEPTION("Executor::Launch: No implementation available for " + backend_name);
    return 1;
  }
};

#ifdef VECCORE_CUDA
/** @brief Specialization of Executor for the CUDA backend */
template <>
class Executor<BackendType::CUDA> : public ExecutorBase<BackendType::CUDA> {
private:
  int fNumSMs;     ///< number of streaming multi-processors
  int *fNelements; ///< device pointer to the number of elements

public:
  Executor(Stream_t stream = 0) : ExecutorBase(stream)
  {
    COPCORE_CUDA_CHECK(cudaMallocManaged(&fNelements, sizeof(int)));
    cudaGetDevice(&fDeviceId);
    cudaDeviceGetAttribute(&fNumSMs, cudaDevAttrMultiProcessorCount, fDeviceId);
  }

  template <class DeviceFunctionPtr, class... Args>
  int Launch(DeviceFunctionPtr func, int n_elements, LaunchGrid_t grid, const Args &... args) const
  {
    constexpr unsigned int warpsPerSM = 32; // we should target a reasonable occupancy
    constexpr unsigned int block_size = 256;

    // Compute the launch grid.
    if (!n_elements) return 0;
    *fNelements = n_elements;
    // cudaMemcpy(fNelements, &n_elements, sizeof(int), cudaMemcpyHostToDevice);

    // Adjust automatically the execution grid. Optimal occupancy:
    // nMaxThreads = fNumSMs * warpsPerSM * 32; grid_size = nmaxthreads / block_size
    // if n_elements < nMaxThreads we reduce the grid size to minimize null threads
    LaunchGrid_t exec_grid{grid};
    if (grid[1].x == 0) {
      unsigned int grid_size =
          std::min(warpsPerSM * fNumSMs * 32 / block_size, (n_elements + block_size - 1) / block_size);
      exec_grid[0].x = grid_size;
      exec_grid[1].x = block_size;
      std::cout << "grid_size = " << grid_size << "  block_size = " << block_size << std::endl;
    }

    // pack parameter addresses into an array
    void *parameter_array[] = {(int *)(&fNelements), const_cast<DeviceFunctionPtr *>(&func),
                               const_cast<Args *>(&args)...};

    // Get a pointer to the kernel implementation global function
    void *kernel_ptr = reinterpret_cast<void *>(&kernel_executor_impl::kernel_dispatch<DeviceFunctionPtr, Args...>);

    // launch the kernel
    COPCORE_CUDA_CHECK(cudaLaunchKernel(kernel_ptr, exec_grid[0], exec_grid[1], parameter_array, 0, fStream));
    return 0;
  }
}; // End  class Executor<BackendType::CUDA>
#endif

/** @brief Specialization of Executor for the CPU backend */
template <>
class Executor<BackendType::CPU> : public ExecutorBase<BackendType::CPU> {
public:
  Executor(Stream_t stream = 0) : ExecutorBase(stream) {}

  template <class HostFunctionPtr, class... Args>
  int Launch(HostFunctionPtr func, int n_elements, LaunchGrid_t /*grid*/, const Args &... args) const
  {
#pragma omp parallel for
    for (int i = 0; i < n_elements; ++i) {
      func(i, args...);
    }
    return 0;
  }
}; // End class Executor<BackendType::CPU>

} // End namespace copcore

#endif // COPCORE_EXECUTOR_H_
