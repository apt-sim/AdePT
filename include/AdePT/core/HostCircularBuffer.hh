// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_HOST_CIRCULAR_BUFFER_HH
#define ADEPT_HOST_CIRCULAR_BUFFER_HH

#include <AdePT/core/GPUStep.hh>

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace AsyncAdePT {

/// @brief Tracks occupied ranges in the pinned host buffer used for returned GPU steps.
/// @details
/// The HostCircularBuffer manages the memory of the pinned HostBuffer in returned-step transfer. It keeps track of
/// the used space in the sorted vector of segments fSegments. The StepProcessingThread adds segments when it submits
/// items to the StepQueue (in fact, the memory is already allocated as soon as the copy from TransferStepsToHost is
/// done). The StepProcessingThread can delete segments when it copies the steps to the holdoutBuffer and the G4Worker
/// finish their work. Since both StepProcessingThread and G4Workers can change the segments, a mutex is used to lock
/// the access to fSegments. In TransferStepsToHost, the StepProcessingThread checks whether there is enough contiguous
/// memory in the HostCircularBuffer before the copy can start.
class HostCircularBuffer {
public:
  struct Segment {
    GPUStep *begin;
    GPUStep *end;

    bool operator<(const Segment &other) const { return begin < other.begin; }
  };

  HostCircularBuffer(GPUStep *bufferStart, std::size_t capacity);

  /// Adds a segment at the provided position. Ensures segments remain sorted.
  bool addSegment(GPUStep *begin, GPUStep *end);

  /// Removes a segment based on its starting pointer and updates fWritePtr accordingly.
  void removeSegment(GPUStep *segmentPtr);

  /// Returns the contiguous free space in front of the fWritePtr (or at the beginning of the buffer in case of a
  /// wraparound).
  std::size_t getFreeContiguousMemory(std::size_t transferSize);

  /// @brief Return the current fill fraction according to the existing transfer-manager bookkeeping.
  /// @details This intentionally preserves the historical semantics: the value is based on fFreeContiguousSpace,
  /// which is updated by getFreeContiguousMemory(transferSize) and includes the pending transfer size.
  double getFillFraction() const;

  std::size_t getOffset() const;

private:
  bool checkForOverlaps() const;
  void printSegments(const std::string &msg) const;

  GPUStep *fBufferStart;
  GPUStep *fBufferEnd;
  GPUStep *fWritePtr;
  std::size_t fFreeContiguousSpace;
  std::vector<Segment> fSegments; // **Sorted vector instead of set**
  mutable std::mutex bufferManagerMutex;
};

} // namespace AsyncAdePT

#endif
