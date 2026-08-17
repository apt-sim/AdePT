// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_track_block.cu
 * @brief Unit tests for the BlockData concurrent container.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <AdePT/transport/containers/BlockData.h>
#include <AdePT/transport/support/Portability.hh>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

namespace {

struct TestTrack {
  int index{0};
  double position[3]{0};
  double direction[3]{0};
  bool flag1{false};
  bool flag2{false};
};

using Block = adept::BlockData<TestTrack>;

struct BlockDeleter {
  void operator()(Block *block) const { Block::ReleaseInstance(block); }
};

using BlockPtr = std::unique_ptr<Block, BlockDeleter>;

class ManagedBlock {
public:
  explicit ManagedBlock(int capacity)
  {
    const auto result = ADEPT_DEVICE_API_SYMBOL(MallocManaged)(&fStorage, Block::SizeOfInstance(capacity));
    if (result != ADEPT_DEVICE_API_SYMBOL(Success)) {
      throw std::runtime_error{ADEPT_DEVICE_API_SYMBOL(GetErrorString)(result)};
    }
    fBlock = Block::MakeInstanceAt(capacity, fStorage);
  }

  ~ManagedBlock()
  {
    if (fBlock) Block::ReleaseInstance(fBlock);
    if (fStorage) ADEPT_DEVICE_API_SYMBOL(Free)(fStorage);
  }

  Block *get() const { return fBlock; }

private:
  char *fStorage{nullptr};
  Block *fBlock{nullptr};
};

__global__ void AcquireTracks(Block *block, unsigned int numAttempts)
{
  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= numAttempts) return;

  auto *track = block->NextElement();
  if (track) track->index = static_cast<int>(id);
}

__global__ void ReleaseTracks(Block *block, unsigned int numToRelease)
{
  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < numToRelease) block->ReleaseElement(static_cast<int>(id));
}

TEST(BlockDataTest, AllocatesUpToCapacityAndReusesReleasedElementsOnHost)
{
  constexpr int capacity = 1024;
  BlockPtr block{Block::MakeInstance(capacity)};
  ASSERT_NE(nullptr, block);
  EXPECT_EQ(capacity, block->Capacity());
  EXPECT_EQ(0, block->GetNused());
  EXPECT_EQ(0, block->GetNholes());

  for (int index = 0; index < capacity; ++index) {
    auto *track = block->NextElement();
    ASSERT_NE(nullptr, track);
    track->index = index;
  }

  EXPECT_TRUE(block->IsFull());
  EXPECT_EQ(nullptr, block->NextElement());
  EXPECT_EQ(capacity, block->GetNused());

  for (int index = 0; index < 16; ++index)
    block->ReleaseElement(index);

  EXPECT_EQ(capacity - 16, block->GetNused());
  EXPECT_EQ(16, block->GetNholes());

  for (int index = 0; index < 16; ++index) {
    auto *track = block->NextElement();
    ASSERT_NE(nullptr, track);
    track->index = capacity + index;
  }

  EXPECT_TRUE(block->IsFull());
  EXPECT_EQ(0, block->GetNholes());
}

TEST(BlockDataTest, RejectsCapacitiesInvalidForNestedQueueWithAssertionsDisabled)
{
  for (const int capacity : {0, 1, 3, 6}) {
    Block *allocated = Block::MakeInstance(capacity);
    EXPECT_EQ(nullptr, allocated) << "capacity " << capacity;
    if (allocated) Block::ReleaseInstance(allocated);

    auto storage  = std::make_unique<char[]>(Block::SizeOfInstance(capacity));
    Block *placed = Block::MakeInstanceAt(capacity, storage.get());
    EXPECT_EQ(nullptr, placed) << "capacity " << capacity;
    if (placed) Block::ReleaseInstance(placed);
  }
}

TEST(BlockDataTest, CopyPreservesStoredValuesAndStartsUndistributed)
{
  constexpr int capacity = 1024;
  BlockPtr source{Block::MakeInstance(capacity)};
  ASSERT_NE(nullptr, source);

  unsigned long long expectedChecksum = 0;
  for (int index = 0; index < capacity; ++index) {
    auto *track = source->NextElement();
    ASSERT_NE(nullptr, track);
    track->index = index;
    expectedChecksum += index;
  }

  auto storage = std::make_unique<char[]>(Block::SizeOfInstance(capacity));
  Block *copy  = Block::MakeCopyAt(*source, storage.get());
  ASSERT_NE(nullptr, copy);
  EXPECT_EQ(0, copy->GetNused());
  EXPECT_EQ(0, copy->GetNholes());

  unsigned long long actualChecksum = 0;
  for (int index = 0; index < capacity; ++index) {
    auto *track = copy->NextElement();
    ASSERT_NE(nullptr, track);
    actualChecksum += track->index;
  }

  EXPECT_EQ(expectedChecksum, actualChecksum);
  Block::ReleaseInstance(copy);
}

TEST(BlockDataTest, ShrinkingCopyDiscardsSourceHolesAndStartsUndistributed)
{
  BlockPtr source{Block::MakeInstance(8)};
  ASSERT_NE(nullptr, source);

  for (int index = 0; index < source->Capacity(); ++index) {
    auto *track = source->NextElement();
    ASSERT_NE(nullptr, track);
    track->index = index;
  }
  source->ReleaseElement(7);
  ASSERT_EQ(1, source->GetNholes());

  auto storage = std::make_unique<char[]>(Block::SizeOfInstance(4));
  Block *copy  = Block::MakeCopyAt(4, *source, storage.get());
  ASSERT_NE(nullptr, copy);
  EXPECT_EQ(4, copy->Capacity());
  EXPECT_EQ(0, copy->GetNused());
  EXPECT_EQ(0, copy->GetNholes());

  for (int index = 0; index < copy->Capacity(); ++index) {
    auto *track = copy->NextElement();
    ASSERT_EQ(&(*copy)[index], track);
    EXPECT_EQ(index, track->index);
  }

  EXPECT_EQ(nullptr, copy->NextElement());
  EXPECT_TRUE(copy->IsFull());
  Block::ReleaseInstance(copy);
}

TEST(BlockDataTest, ConcurrentDeviceAllocationReleaseAndReuse)
{
  constexpr unsigned int capacity       = 1u << 16;
  constexpr unsigned int initialTracks  = 1u << 14;
  constexpr unsigned int releasedTracks = 1u << 11;
  constexpr unsigned int refillTracks   = 1u << 10;
  constexpr dim3 threads{128};

  ManagedBlock block{capacity};

  AcquireTracks<<<initialTracks / threads.x, threads>>>(block.get(), initialTracks);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
  ASSERT_EQ(initialTracks, static_cast<unsigned int>(block.get()->GetNused()));

  unsigned long long actualChecksum = 0;
  for (unsigned int index = 0; index < initialTracks; ++index)
    actualChecksum += (*block.get())[index].index;
  const auto expectedChecksum = static_cast<unsigned long long>(initialTracks) * (initialTracks - 1) / 2;
  EXPECT_EQ(expectedChecksum, actualChecksum);

  ReleaseTracks<<<releasedTracks / threads.x, threads>>>(block.get(), releasedTracks);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
  EXPECT_EQ(initialTracks - releasedTracks, static_cast<unsigned int>(block.get()->GetNused()));
  EXPECT_EQ(releasedTracks, static_cast<unsigned int>(block.get()->GetNholes()));

  AcquireTracks<<<refillTracks / threads.x, threads>>>(block.get(), refillTracks);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
  EXPECT_EQ(initialTracks - releasedTracks + refillTracks, static_cast<unsigned int>(block.get()->GetNused()));
  EXPECT_EQ(releasedTracks - refillTracks, static_cast<unsigned int>(block.get()->GetNholes()));
}

} // namespace
