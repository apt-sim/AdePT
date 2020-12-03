// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef COPCORE_RANLUXPP_H_
#define COPCORE_RANLUXPP_H_

#include <CopCore/backend/BackendCommon.h>

#include "mulmod.h"

#include <cassert>
#include <cstdint>

namespace {

__device__
const uint64_t kA_2048[] = {
    0xed7faa90747aaad9, 0x4cec2c78af55c101, 0xe64dcb31c48228ec, 0x6d8a15a13bee7cb0, 0x20b2ca60cb78c509,
    0x256c3d3c662ea36c, 0xff74e54107684ed2, 0x492edfcc0cc8e753, 0xb48c187cf5b22097,
};

} // end anonymous namespace

template <int w>
class RanluxppEngineImpl {

private:
  uint64_t fState[9]; ///< State of the generator
  int fPosition = 0;  ///< Current position in bits

  static constexpr const uint64_t *kA = kA_2048;
  static constexpr int kMaxPos        = 9 * 64;

  /// Produce next block of random bits
  __host__ __device__
  void Advance()
  {
    mulmod(kA, fState);
    fPosition = 0;
  }

public:
  RanluxppEngineImpl() = default;

  /// Return the next random bits, generate a new block if necessary
  __host__ __device__
  uint64_t NextRandomBits()
  {
    if (fPosition + w > kMaxPos) {
      Advance();
    }

    int idx     = fPosition / 64;
    int offset  = fPosition % 64;
    int numBits = 64 - offset;

    uint64_t bits = fState[idx] >> offset;
    if (numBits < w) {
      bits |= fState[idx + 1] << numBits;
    }
    bits &= ((uint64_t(1) << w) - 1);

    fPosition += w;
    assert(fPosition <= kMaxPos && "position out of range!");

    return bits;
  }

  /// Initialize and seed the state of the generator
  __host__ __device__
  void SetSeed(uint64_t s)
  {
    fState[0] = 1;
    for (int i = 1; i < 9; i++) {
      fState[i] = 0;
    }

    uint64_t a_seed[9];
    // Skip 2 ** 96 states.
    powermod(kA, a_seed, uint64_t(1) << 48);
    powermod(a_seed, a_seed, uint64_t(1) << 48);
    // Skip another s states.
    powermod(a_seed, a_seed, s);
    mulmod(a_seed, fState);

    fPosition = 0;
  }

  /// Skip `n` random numbers without generating them
  __host__ __device__
  void Skip(uint64_t n)
  {
    int left = (kMaxPos - fPosition) / w;
    assert(left >= 0 && "position was out of range!");
    if (n < (uint64_t)left) {
      // Just skip the next few entries in the currently available bits.
      fPosition += n * w;
      assert(fPosition <= kMaxPos && "position out of range!");
      return;
    }

    n -= left;
    // Need to advance and possibly skip over blocks.
    int nPerState = kMaxPos / w;
    int skip      = (n / nPerState);

    uint64_t a_skip[9];
    powermod(kA, a_skip, skip + 1);
    mulmod(a_skip, fState);

    // Potentially skip numbers in the freshly generated block.
    int remaining = n - skip * nPerState;
    assert(remaining >= 0 && "should not end up at a negative position!");
    fPosition = remaining * w;
    assert(fPosition <= kMaxPos && "position out of range!");
  }
};

class RanluxppDouble : public RanluxppEngineImpl<52> {
public:
  __host__ __device__
  RanluxppDouble(uint64_t seed = 314159265) { this->SetSeed(seed); }

  __host__ __device__
  double Rndm() { return (*this)(); }

  __host__ __device__
  double operator()()
  {
    // Get 52 bits of randomness.
    uint64_t bits = this->NextRandomBits();

    // Construct the double in [1, 2), using the random bits as mantissa.
    static constexpr uint64_t exp = 0x3ff0000000000000;
    union {
      double dRandom;
      uint64_t iRandom;
    };
    iRandom = exp | bits;

    // Shift to the right interval of [0, 1).
    return dRandom - 1;
  }

  __host__ __device__
  uint64_t IntRndm() { return this->NextRandomBits(); }
};

#endif // COPCORE_RANLUXPP_H_
