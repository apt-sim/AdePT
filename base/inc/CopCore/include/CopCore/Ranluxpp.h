// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: LGPL-2.1-or-later

#ifndef COPCORE_RANLUXPP_H_
#define COPCORE_RANLUXPP_H_

#include <CopCore/Global.h>

#include "ranluxpp/mulmod.h"
#include "ranluxpp/ranlux_lcg.h"

#include <cassert>
#include <cstdint>

namespace {

__device__ const uint64_t kA_2048[] = {
    0xed7faa90747aaad9, 0x4cec2c78af55c101, 0xe64dcb31c48228ec, 0x6d8a15a13bee7cb0, 0x20b2ca60cb78c509,
    0x256c3d3c662ea36c, 0xff74e54107684ed2, 0x492edfcc0cc8e753, 0xb48c187cf5b22097,
};

} // end anonymous namespace

template <int w>
class RanluxppEngineImpl {

private:
  uint64_t fState[9]; ///< RANLUX state of the generator
  unsigned fCarry;    ///< Carry bit of the RANLUX state
  int fPosition = 0;  ///< Current position in bits

  static constexpr const uint64_t *kA = kA_2048;
  static constexpr int kMaxPos        = 9 * 64;

protected:
  __host__ __device__
  void SaveState(uint64_t *state) const
  {
    for (int i = 0; i < 9; i++) {
      state[i] = fState[i];
    }
  }

  __host__ __device__
  void XORstate(const uint64_t *state)
  {
    for (int i = 0; i < 9; i++) {
      fState[i] ^= state[i];
    }
  }

public:
  RanluxppEngineImpl() = default;

  /// Produce next block of random bits
  __host__ __device__ void Advance()
  {
    uint64_t lcg[9];
    to_lcg(fState, fCarry, lcg);
    mulmod(kA, lcg);
    to_ranlux(lcg, fState, fCarry);
    fPosition = 0;
  }

  /// Return the next random bits, generate a new block if necessary
  __host__ __device__ uint64_t NextRandomBits()
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

  /// Return a floating point number, converted from the next random bits.
  __host__ __device__ double NextRandomFloat()
  {
    static constexpr double div = 1.0 / (uint64_t(1) << w);
    uint64_t bits               = NextRandomBits();
    return bits * div;
  }

  /// Initialize and seed the state of the generator
  __host__ __device__ void SetSeed(uint64_t s)
  {
    uint64_t lcg[9];
    lcg[0] = 1;
    for (int i = 1; i < 9; i++) {
      lcg[i] = 0;
    }

    uint64_t a_seed[9];
    // Skip 2 ** 96 states.
    powermod(kA, a_seed, uint64_t(1) << 48);
    powermod(a_seed, a_seed, uint64_t(1) << 48);
    // Skip another s states.
    powermod(a_seed, a_seed, s);
    mulmod(a_seed, lcg);

    to_ranlux(lcg, fState, fCarry);
    fPosition = 0;
  }

  /// Skip `n` random numbers without generating them
  __host__ __device__ void Skip(uint64_t n)
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

    uint64_t lcg[9];
    to_lcg(fState, fCarry, lcg);
    mulmod(a_skip, lcg);
    to_ranlux(lcg, fState, fCarry);

    // Potentially skip numbers in the freshly generated block.
    int remaining = n - skip * nPerState;
    assert(remaining >= 0 && "should not end up at a negative position!");
    fPosition = remaining * w;
    assert(fPosition <= kMaxPos && "position out of range!");
  }
};

class RanluxppDouble final : public RanluxppEngineImpl<48> {
public:
  __host__ __device__ RanluxppDouble(uint64_t seed = 314159265) { this->SetSeed(seed); }

  /// Generate a double-precision random number with 48 bits of randomness
  __host__ __device__ double Rndm() { return (*this)(); }
  /// Generate a double-precision random number (non-virtual method)
  __host__ __device__ double operator()() { return this->NextRandomFloat(); }
  /// Generate a random integer value with 48 bits
  __host__ __device__ uint64_t IntRndm() { return this->NextRandomBits(); }

  /// Branch a new RNG state, also advancing the current one.
  /// The caller must Advance() the branched RNG state to decorrelate the
  /// produced numbers.
  __host__ __device__ RanluxppDouble BranchNoAdvance()
  {
    // Save the current state, will be used to branch a new RNG.
    uint64_t oldState[9];
    this->SaveState(oldState);
    this->Advance();
    // Copy and modify the new RNG state.
    RanluxppDouble newRNG(*this);
    newRNG.XORstate(oldState);
    return newRNG;
  }

  /// Branch a new RNG state, also advancing the current one.
  __host__ __device__ RanluxppDouble Branch()
  {
    RanluxppDouble newRNG(BranchNoAdvance());
    newRNG.Advance();
    return newRNG;
  }
};

#endif
