// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ADEPT_HOST_CIRCULAR_BUFFER_HH
#define ADEPT_HOST_CIRCULAR_BUFFER_HH

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
    std::size_t begin{0};
    std::size_t end{0};

    bool operator<(const Segment &other) const { return begin < other.begin; }
  };

  explicit HostCircularBuffer(std::size_t capacity);

  /// Adds a segment at the provided position. Ensures segments remain sorted.
  bool AddSegment(std::size_t begin, std::size_t end);

  /// Removes a segment based on its starting offset.
  void RemoveSegment(std::size_t segmentBegin);

  /// Returns the contiguous free space in front of the write offset (or at the beginning of the buffer in case of a
  /// wraparound).
  std::size_t GetFreeContiguousSlots(std::size_t transferSize);

  /// @brief Return the fill fraction according to the existing transfer-manager bookkeeping.
  /// @details This intentionally preserves the historical semantics: the value is based on fFreeContiguousSpace,
  /// which is updated by GetFreeContiguousSlots(transferSize) and includes the pending transfer size.
  double GetFillFractionAfterLastRequest() const;

  std::size_t GetWriteOffset() const;

private:
  bool checkForOverlaps() const;
  void printSegments(const std::string &msg) const;

  std::size_t fCapacity{0};
  std::size_t fWriteOffset{0};
  std::size_t fFreeContiguousSpace;
  std::vector<Segment> fSegments; // sorted by segment start offset
  mutable std::mutex bufferManagerMutex;
};

} // namespace AsyncAdePT

#endif
