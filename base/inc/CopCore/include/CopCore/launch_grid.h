// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file launch_grid.h
 * @brief A kernel launch grid of blocks/threads.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_LAUNCH_GRID_H_
#define ADEPT_LAUNCH_GRID_H_

#include "Global.h"

namespace copcore {

/** @brief Helper allowing to handle kernel launch configurations as compact type */
template <BackendType backend>
class launch_grid {
};

template <>
class launch_grid<BackendType::CPU> {
private:
  int fGrid[2]; ///< Number of blocks/threads per block

public:
  launch_grid(int n_blocks, int n_threads) : fGrid{n_blocks, n_threads} {}

  /** @brief Access either block [0] or thread [1] grid */
  __host__ __device__
  int operator[](int index) { return fGrid[index]; }

  /** @brief Access either block [0] or thread [1] grid */
  __host__ __device__
  int operator[](int index) const { return fGrid[index]; }

}; // End class launch_grid<BackendType::CPU>

#ifdef TARGET_DEVICE_CUDA
template <>
class launch_grid<BackendType::CUDA> {
private:
  dim3 fGrid[2]; ///< Block and thread index grid

public:
  /** @brief Construct from block and thread grids */
  __host__ __device__
  launch_grid(const dim3 &block_index, const dim3 &thread_index) : fGrid{block_index, thread_index} {}

  /** @brief Default constructor */
  __host__ __device__
  launch_grid() : launch_grid(dim3(), dim3()) {}

  /** @brief Access either block [0] or thread [1] grid */
  __host__ __device__
  dim3 &operator[](int index) { return fGrid[index]; }

  /** @brief Access either block [0] or thread [1] grid */
  __host__ __device__
  const dim3 &operator[](int index) const { return fGrid[index]; }
}; // End class launch_grid<BackendType::CUDA>
#endif

} // End namespace copcore

#endif // ADEPT_LAUNCH_GRID_H_
