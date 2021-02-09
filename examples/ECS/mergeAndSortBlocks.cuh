#ifndef MERGE_AND_SORT_BLOCKS_H
#define MERGE_AND_SORT_BLOCKS_H

#include "ecs.h"

template<unsigned int N, unsigned int M>
__host__ __device__
unsigned int occupancy(TrackBlock<N> tb[M]) {
  unsigned int acc = 0;
  for (unsigned int i = 0; i < M; ++i) {
    for (const auto id : tb[i].id) {
      if (id >= 0)
        acc++;
    }
  }
  
  return acc;
}


template<unsigned int TBSize, unsigned int ArrSize>
__device__ void merge_blocks(TrackBlock<TBSize> a[ArrSize], TrackBlock<TBSize> b[ArrSize]) {
  constexpr bool debugPrint = false;
  constexpr auto N = TBSize * ArrSize;
  static_assert(TBSize == 1 || ArrSize == 1, "Only support merging across one dimension");
  
  __shared__ int32_t ids_a[N];
  __shared__ int32_t ids_b[N];
  __shared__ int32_t listOfHoles[N];
  __shared__ int32_t destinationIdx[N];
  __shared__ int32_t counter_a;
  __shared__ int32_t counter_b;

  const auto occupa = occupancy<TBSize, ArrSize>(a);
  const auto occupb = occupancy<TBSize, ArrSize>(b);

  if (a == b || occupb == 0 || occupb > N - occupa)
    return;

  // Initialise all work arrays, and load ids to check if particles are alive
  for (unsigned int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    const auto arrIdx = tid % ArrSize;
    const auto tbIdx  = tid % TBSize;
    ids_a[tid] = a[arrIdx].id[tbIdx];
    ids_b[tid] = b[arrIdx].id[tbIdx];
    listOfHoles[tid] = -1;
    destinationIdx[tid] = -1;
  }
  counter_a = 0;
  counter_b = 0;
  __syncthreads();

  
  // Search holes in a
  for (unsigned int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    if (ids_a[tid] < 0) {
      const auto counter = atomicAdd(&counter_a, 1);
      listOfHoles[counter] = tid;
    }
  }
  __syncthreads();
  

  // Determine where b's elements go in a
  for (unsigned int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    if (ids_b[tid] >= 0) {
      const auto counter = atomicAdd(&counter_b, 1);
      destinationIdx[tid] = listOfHoles[counter];
    } else {
      destinationIdx[tid] = -1;
    }
  }
  __syncthreads();


  // If an element should move, move it now
  for (unsigned int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    if (destinationIdx[tid] >= 0) {
      const auto dest = destinationIdx[tid];
      const auto src  = tid;
      assert(dest < N);
      assert(src < N);

      if (debugPrint && (threadIdx.x < 3 && blockIdx.x < 2))
        printf("(%d %d) Moving (%d id=%d) to (%d id=%d)\n", blockIdx.x, threadIdx.x, src, ids_b[src], dest, a[0].id[dest]);

      if constexpr (ArrSize == 1) {
        a[0][dest] = b[0][src];
      } else {
        static_assert(TBSize == 1, "No merging across two dimensions.");
        memcpy(&a[dest], &b[src], sizeof(TrackBlock<TBSize>));
      }

      // Mask the element in b
      b[src % ArrSize].id[src % TBSize] = -1 * abs(b[src % ArrSize].id[src % TBSize]);
    }
  }


#ifndef NDEBUG
  __syncthreads();
  const auto newOccupb = occupancy<TBSize, ArrSize>(b);
  const auto newOccupa = occupancy<TBSize, ArrSize>(a);
  if (debugPrint && threadIdx.x == 0) {
    printf("a= ");
    for (unsigned int i = 0; i < N; ++i) {
      if (a[i%ArrSize].id[i%TBSize] != ids_a[i])
        printf(" %d=%d\t(was %d)\n", i, a[i % ArrSize].id[i % TBSize], ids_a[i]);
    }
    printf("\nb=");
    for (unsigned int i = 0; i < N; ++i) {
      if (ids_b[i] >= 0) {
        bool found = false;
        for (unsigned int j = 0; j < N; ++j) {
          if (ids_b[i] == a[j%ArrSize].id[j%TBSize])
            found = true;
        }
        if (!found)
          printf("!!!! Not found %d ", ids_b[i]);
      }
      printf(" %d=%d\n", i, b[i % ArrSize].id[i % TBSize]);
    }
    printf("\n");
  }
  __syncthreads();

  assert(newOccupb == 0);
  assert(newOccupa == occupa + occupb);
#endif
}


template<unsigned int TBSize, unsigned int ArrSize>
__device__ void sort_block(TrackBlock<TBSize> a[ArrSize]) {
  constexpr auto N = TBSize * ArrSize;
  static_assert(TBSize == 1 || ArrSize == 1, "Only support sorting in one dimension");

  __shared__ int32_t listOfHoles[N];
  __shared__ int32_t destinationIdx[N];
  __shared__ int32_t counter_a;
  __shared__ int32_t counter_b;
  counter_a = 0;
  counter_b = 0;
  const auto occupa = occupancy<TBSize, ArrSize>(a);
  __syncthreads();

  // Find all holes that are at an index < occupancy
  for (unsigned int tid = threadIdx.x; tid < occupa; tid += blockDim.x) {
    const auto arrIdx = tid % ArrSize;
    const auto tbIdx  = tid % TBSize;
    if (a[arrIdx][tbIdx].id < 0) {
      const auto counter = atomicAdd(&counter_a, 1);
      listOfHoles[counter] = tid;
    }
  }
  __syncthreads();

  // Determine where elements in the back go
  for (unsigned int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    const auto arrIdx = tid % ArrSize;
    const auto tbIdx  = tid % TBSize;
    if (tid >= occupa && a[arrIdx].id[tbIdx] >= 0) {
      const auto counter = atomicAdd(&counter_b, 1);
      destinationIdx[tid] = listOfHoles[counter];
    } else {
      destinationIdx[tid] = -1;
    }
  }
  __syncthreads();

  for (unsigned int tid = threadIdx.x; tid < N; tid += blockDim.x) {
    if (destinationIdx[tid] >= 0) {
      const auto dest = destinationIdx[tid];
      const auto src  = tid;
      assert(dest < occupa);
      assert(src >= occupa);

      if constexpr (ArrSize == 1) {
        assert(a[0][dest].id < 0);
        assert(a[0][src ].id >= 0);
        a[0][dest] = a[0][src];
      } else {
        static_assert(TBSize == 1, "No sorting in two dimensions.");
        memcpy(&a[dest], &a[src], sizeof(TrackBlock<TBSize>));
      }

      a[src % ArrSize][src % TBSize].id = -1 * abs(a[src % ArrSize][src % TBSize].id);
    }
  }

#ifndef NDEBUG
  __syncthreads();
  const auto newOccupa = occupancy<TBSize, ArrSize>(a);
  assert(newOccupa == occupa);
#endif
}


template<typename Slab, bool sort = false>
__device__ void compactOrSort_blocks(Slab* slab)
{
  for (unsigned int taskIdx = 0; taskIdx < Slab::tasks_per_slab; ++taskIdx) {
    for (unsigned int bIdx = blockIdx.x; bIdx < SlabSoA::blocks_per_task; bIdx += gridDim.x) {

      if constexpr (sort) {
        if constexpr (Slab::tracks_per_block > 1) {
          sort_block<Slab::tracks_per_block, 1>(
              &slab->tracks[taskIdx][bIdx]);
        } else {
          sort_block<1, SlabSoA::tracks_per_block>(
              &slab->tracks[taskIdx][bIdx * SlabSoA::tracks_per_block]);
        }
      } else {
        constexpr unsigned int targetTaskIdx = 0;
        assert(taskIdx < Slab::tasks_per_slab);
        if constexpr (Slab::tracks_per_block > 1) {
          // Directly merge a block of tracks into another
          assert(bIdx < Slab::blocks_per_task);
          assert(taskIdx < Slab::tasks_per_slab);
          merge_blocks<Slab::tracks_per_block, 1>(
              &slab->tracks[targetTaskIdx][bIdx],
              &slab->tracks[taskIdx][bIdx]);
        } else {
          // Because of AoS layout, merge an array of the size of the SoA layout into another
          const auto blockIdxSoA = bIdx * SlabSoA::tracks_per_block;
          assert(blockIdxSoA < Slab::blocks_per_task);
          merge_blocks<1, SlabSoA::tracks_per_block>(
              &slab->tracks[targetTaskIdx][blockIdxSoA],
              &slab->tracks[taskIdx][blockIdxSoA]);
        }
      }
    }
  }
}

template<typename Slab>
__global__ void run_merge_blocks(Slab* slab)
{
  compactOrSort_blocks<Slab, false>(slab);
}

template<typename Slab>
__global__ void run_sort_blocks(Slab* slab)
{
  compactOrSort_blocks<Slab, true>(slab);
}

#endif