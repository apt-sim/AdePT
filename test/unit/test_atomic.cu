// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_atomic.cu
 * @brief Unit tests for host and device atomic operations.
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#include <AdePT/transport/containers/Atomic.h>
#include <AdePT/transport/support/Portability.hh>

#include <gtest/gtest.h>

#include <new>

namespace {

struct SomeStruct {
  adept::Atomic_t<int> varInt;
  adept::Atomic_t<float> varFloat;

  __host__ __device__ SomeStruct() {}
};

__global__ void AddValues(SomeStruct *values)
{
  values->varInt.fetch_add(1);
  values->varFloat.fetch_add(1);
}

__global__ void SubtractValues(SomeStruct *values)
{
  values->varInt.fetch_sub(1);
  values->varFloat.fetch_sub(1);
}

__global__ void CycleCompareExchange(SomeStruct *values)
{
  auto expected = values->varInt.load();
  bool success  = false;
  while (!success) {
    while (expected > 0) {
      success = values->varInt.compare_exchange_strong(expected, expected - 1);
      if (success) return;
    }
    while (expected == 0) {
      success = values->varInt.compare_exchange_strong(expected, 100);
      if (success) return;
    }
  }
}

__global__ void AssignAtomicBase(adept::AtomicBase_t<int> *destination, const adept::AtomicBase_t<int> *source,
                                 bool *returnedDestination)
{
  auto &result         = (*destination = *source);
  *returnedDestination = &result == destination;
}

template <typename T>
T *MakeManagedObject()
{
  T *object         = nullptr;
  const auto result = ADEPT_DEVICE_API_SYMBOL(MallocManaged)(&object, sizeof(T));
  if (result != ADEPT_DEVICE_API_SYMBOL(Success)) {
    throw std::runtime_error{ADEPT_DEVICE_API_SYMBOL(GetErrorString)(result)};
  }
  return new (object) T{};
}

template <typename T>
void ReleaseManagedObject(T *object)
{
  object->~T();
  ADEPT_DEVICE_API_CALL(Free(object));
}

TEST(AtomicTest, CopyAssignmentReturnsDestinationOnHost)
{
  adept::AtomicBase_t<int> source;
  adept::AtomicBase_t<int> destination;
  source.store(42);
  destination.store(7);

  auto &result = (destination = source);

  EXPECT_EQ(&destination, &result);
  EXPECT_EQ(42, destination.load());

  adept::Atomic_t<int> derivedSource;
  adept::Atomic_t<int> derivedDestination;
  derivedSource.store(73);

  auto &derivedResult = (derivedDestination = derivedSource);

  EXPECT_EQ(&derivedDestination, &derivedResult);
  EXPECT_EQ(73, derivedDestination.load());
}

TEST(AtomicTest, CopyAssignmentReturnsDestinationOnDevice)
{
  auto *source      = MakeManagedObject<adept::AtomicBase_t<int>>();
  auto *destination = MakeManagedObject<adept::AtomicBase_t<int>>();
  auto *returned    = MakeManagedObject<bool>();
  source->store(1234);
  destination->store(-1);
  *returned = false;

  AssignAtomicBase<<<1, 1>>>(destination, source, returned);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  EXPECT_TRUE(*returned);
  EXPECT_EQ(1234, destination->load());

  ReleaseManagedObject(returned);
  ReleaseManagedObject(destination);
  ReleaseManagedObject(source);
}

TEST(AtomicTest, ConcurrentAdditionAndSubtraction)
{
  constexpr dim3 blocks{256};
  constexpr dim3 threads{128};
  constexpr int numOperations = blocks.x * threads.x;

  auto *values = MakeManagedObject<SomeStruct>();

  AddValues<<<blocks, threads>>>(values);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
  EXPECT_EQ(numOperations, values->varInt.load());
  EXPECT_FLOAT_EQ(static_cast<float>(numOperations), values->varFloat.load());

  SubtractValues<<<blocks, threads>>>(values);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());
  EXPECT_EQ(0, values->varInt.load());
  EXPECT_FLOAT_EQ(0.0f, values->varFloat.load());

  ReleaseManagedObject(values);
}

TEST(AtomicTest, ConcurrentCompareExchange)
{
  auto *values = MakeManagedObject<SomeStruct>();
  values->varInt.store(99);

  // A 101-operation cycle decrements 99 to zero, resets it to 100, and
  // decrements it once more to the original value.
  CycleCompareExchange<<<1, 101>>>(values);
  ASSERT_EQ(ADEPT_DEVICE_API_SYMBOL(Success), ADEPT_DEVICE_API_SYMBOL(DeviceSynchronize)());

  EXPECT_EQ(99, values->varInt.load());
  ReleaseManagedObject(values);
}

} // namespace
