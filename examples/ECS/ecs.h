#ifndef EXAMPLES_ECS_ECS_H_
#define EXAMPLES_ECS_ECS_H_

#ifndef __CUDACC__
#  define __host__
#  define __device__
#  define __global__
#endif

#include <CopCore/Ranluxpp.h>

#include <array>
#include <cstddef>
#include <cstdint>

/**
 * This is a SoA that's holding track data. If NumberOfElements == 1, it collapses to a single track.
 * It's meant to run as 1 block on the GPU, so we assign something like 128, 256, [...] threads to process it.
 */
template<unsigned int NumberOfElements>
struct TrackBlock {
  static constexpr unsigned int nSlot = NumberOfElements;

  /* event structure */

  int32_t id[nSlot];
  int32_t parent[nSlot];

  /* geometry data */

  float x[nSlot];
  float y[nSlot];
  float z[nSlot];

  int32_t geometry_id[nSlot];

  /* physics data */

  float vx[nSlot];
  float vy[nSlot];
  float vz[nSlot];
  float E[nSlot];

  int32_t material_id[nSlot];

  /* time */

  float global_time[nSlot];
  float proper_time[nSlot];

  /* random numbers */

  RanluxppDouble rng_state[nSlot]; // 80 bytes

  // A struct of references to the arrays above. Using this,
  // an entry in the TrackBlock can be processed as if it was
  // a simple struct.
  struct ElementAccessor {
    decltype(id[0])& id;
    decltype(parent[0])& parent;

    decltype(x[0])& x;
    decltype(y[0])& y;
    decltype(z[0])& z;

    decltype(geometry_id[0])& geometry_id;

    decltype(vx[0])& vx;
    decltype(vy[0])& vy;
    decltype(vz[0])& vz;
    decltype(E[0])& E;

    decltype(material_id[0])& material_id;

    decltype(global_time[0])& global_time;
    decltype(proper_time[0])& proper_time;

    decltype(rng_state[0])& rng_state;

    __host__ __device__ ElementAccessor& operator=(const ElementAccessor& other) {
      id = other.id;
      parent = other.parent;
      x = other.x;
      y = other.y;
      z = other.z;
      geometry_id = other.geometry_id;
      vx = other.vx;
      vy = other.vy;
      vz = other.vz;
      E = other.E;
      material_id = other.material_id;
      global_time = other.global_time;
      proper_time = other.proper_time;
      rng_state = other.rng_state;

      return *this;
    }
  };
  using element_type = ElementAccessor;

  // Same as above, but all const references.
  struct ConstElementAccessor {
    const decltype(id[0])& id;
    const decltype(parent[0])& parent;

    const decltype(x[0])& x;
    const decltype(y[0])& y;
    const decltype(z[0])& z;

    const decltype(geometry_id[0])& geometry_id;

    const decltype(vx[0])& vx;
    const decltype(vy[0])& vy;
    const decltype(vz[0])& vz;
    const decltype(E[0])& E;

    const decltype(material_id[0])& material_id;

    const decltype(global_time[0])& global_time;
    const decltype(proper_time[0])& proper_time;

    const decltype(rng_state[0])& rng_state;

    ConstElementAccessor& operator=(const ConstElementAccessor& other) = default;
  };
  using const_element_type = ConstElementAccessor;

  __host__ __device__
  element_type operator[](unsigned int i) {
    assert(i < nSlot);
    return ElementAccessor{
      id[i],
      parent[i],
      x[i],
      y[i],
      z[i],
      geometry_id[i],
      vx[i],
      vy[i],
      vz[i],
      E[i],
      material_id[i],
      global_time[i],
      proper_time[i],
      rng_state[i]
    };
  }
  const_element_type operator[](unsigned int i) const {
    return const_cast<TrackBlock*>(this)->operator[](i);
  }

  __host__ __device__
  void dump(unsigned int i) {
    if (i < nSlot) {
      uint64_t* rng = reinterpret_cast<uint64_t*>(&rng_state[i]);
      printf("\tid=%8d x=(%8.4f %8.4f %8.4f) v=(%8.4f %8.4f %8.4f) E=%8.4f rand=%lX %lX ...\n",
          id[i], x[i], y[i], z[i], vx[i], vy[i], vz[i], E[i],
          rng[0], rng[1]);
    }
    else
      printf("Invalid index %d (nSlot=%d)", i, nSlot);
  }
};

// Slab of memory in SoA style.
struct SlabSoA {

#ifdef NDEBUG
  static constexpr int tracks_per_block = 256;
  static constexpr int blocks_per_task  = 512;
  static constexpr int tasks_per_slab   =  10;
#else
  static constexpr int tracks_per_block = 128;
  static constexpr int blocks_per_task  =  10;
  static constexpr int tasks_per_slab   =   5;
#endif

  struct TrackBlock<tracks_per_block> tracks[tasks_per_slab][blocks_per_task];
};

// Slab of memory in AoS style.
struct SlabAoS {
  static constexpr int tracks_per_block = 1;
  static constexpr int blocks_per_task  = SlabSoA::blocks_per_task * SlabSoA::tracks_per_block;
  static constexpr int tasks_per_slab   = SlabSoA::tasks_per_slab;

  struct TrackBlock<tracks_per_block> tracks[tasks_per_slab][blocks_per_task];
};



#endif /* EXAMPLES_ECS_ECS_H_ */
