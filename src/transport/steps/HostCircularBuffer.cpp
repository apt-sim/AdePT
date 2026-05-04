// SPDX-FileCopyrightText: 2026 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/core/HostCircularBuffer.hh>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>

// definitions for printouts and advanced debugging
// #define DEBUG
#define RESET "\033[0m"
#define BOLD_RED "\033[1;31m"

namespace AsyncAdePT {

HostCircularBuffer::HostCircularBuffer(std::size_t capacity)
    : fCapacity(capacity), fWriteOffset(0), fFreeContiguousSpace(capacity)
{
}

bool HostCircularBuffer::AddSegment(std::size_t begin, std::size_t end)
{
  std::scoped_lock lock{bufferManagerMutex};

  // Insert the segment into sorted position
  fSegments.insert(std::upper_bound(fSegments.begin(), fSegments.end(), Segment{begin, end}), {begin, end});

#ifdef DEBUG
  if (begin != fWriteOffset)
    std::cout << BOLD_RED << " Begin != fWriteOffset " << begin << " fWriteOffset " << fWriteOffset << RESET
              << std::endl;
  std::size_t size = end - begin;
  if (size > fFreeContiguousSpace)
    std::cout << BOLD_RED << " Not enough space! size " << size << " fFreeContiguousSpace " << fFreeContiguousSpace
              << RESET << std::endl;
  if (!checkForOverlaps()) std::cout << BOLD_RED << " Overlaps after AddSegment! " << RESET << std::endl;
#endif

  fWriteOffset = end;

  return true;
}

void HostCircularBuffer::RemoveSegment(std::size_t segmentBegin)
{
  std::scoped_lock lock{bufferManagerMutex};

  // Find the segment
  auto it = std::find_if(fSegments.begin(), fSegments.end(),
                         [segmentBegin](const Segment &seg) { return seg.begin == segmentBegin; });

#ifdef DEBUG
  if (it == fSegments.end())
    std::cout << BOLD_RED << " Trying to remove segment that doesn't exist !! segment : " << segmentBegin << RESET
              << std::endl;
  if (!checkForOverlaps()) std::cout << BOLD_RED << " Overlaps after removesegment! " << RESET << std::endl;
#endif

  // delete it from the list
  fSegments.erase(it);
}

std::size_t HostCircularBuffer::GetFreeContiguousSlots(std::size_t transferSize)
{
  std::scoped_lock lock{bufferManagerMutex};

  if (fSegments.empty()) {
    // If empty, reset the write offset to the beginning of the buffer.
    fWriteOffset = 0;
    return fCapacity; // Everything is free
  }

  // Find the next segment after fWriteOffset
  auto nextSegment = std::lower_bound(fSegments.begin(), fSegments.end(), Segment{fWriteOffset, 0},
                                      [](const Segment &a, const Segment &b) { return a.begin < b.begin; });

  // Find the previous segment before nextSegment (if it exists)
  auto prevSegment = (nextSegment != fSegments.begin()) ? std::prev(nextSegment) : fSegments.end();

  // If fWriteOffset was set on an end of a segment that was deleted, we can put it back to the last previous
  // existing segment
  if (prevSegment != fSegments.end() && fWriteOffset != prevSegment->end) {
    fWriteOffset = prevSegment->end;
  }

  // Free space from fWriteOffset to next segment
  std::size_t forwardSpace =
      (nextSegment != fSegments.end()) ? nextSegment->begin - fWriteOffset : fCapacity - fWriteOffset;

  // Free space for a wraparound
  std::size_t wrapAroundSpace = (fSegments.front().begin > 0) ? fSegments.front().begin : 0;

  fFreeContiguousSpace = forwardSpace + wrapAroundSpace - transferSize;

  if (forwardSpace >= transferSize) {
    return forwardSpace; // Enough space in the current region
  }

  if (wrapAroundSpace >= transferSize) {
    fWriteOffset = 0; // Reset fWriteOffset since we need to wrap around
    return wrapAroundSpace;
  }

#ifdef DEBUG
  std::cout << BOLD_RED
            << "Cannot transfer from Device to Host due to lack of space in CPU HostBuffer. This should never be "
               "the case! transfersize "
            << transferSize << " forwardSpace " << forwardSpace << " wraparoundspace " << wrapAroundSpace
            << " total space : " << fCapacity << " free space " << fFreeContiguousSpace << RESET << std::endl;
  checkForOverlaps();
#endif

  return 0; // Not enough contiguous space available
}

double HostCircularBuffer::GetFillFractionAfterLastRequest() const
{
  return 1. - static_cast<double>(fFreeContiguousSpace) / fCapacity;
}

std::size_t HostCircularBuffer::GetWriteOffset() const
{
  return fWriteOffset;
}

bool HostCircularBuffer::checkForOverlaps() const
{
  for (std::size_t i = 1; i < fSegments.size(); i++) {
    if (fSegments[i - 1].end > fSegments[i].begin) {
      std::cerr << "ERROR: Overlapping segments detected!\n";
      std::cerr << " Segment 1: [" << fSegments[i - 1].begin << " - " << fSegments[i - 1].end << "]\n";
      std::cerr << " Segment 2: [" << fSegments[i].begin << " - " << fSegments[i].end << "]\n";
      assert(false && "Overlapping segments in HostCircularBuffer!");
      return false;
    }
  }
  return true;
}

void HostCircularBuffer::printSegments(const std::string &msg) const
{
  std::cout << msg << " | Current segments: ";
  for (const auto &seg : fSegments) {
    std::cout << "[" << seg.begin << " - " << seg.end << "] ";
  }
  std::cout << std::endl;
}

} // namespace AsyncAdePT
