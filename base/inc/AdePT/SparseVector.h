// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

/**
 * @file SparseVector.h
 * @brief Multi-producer variable-size vector adopting pre-allocatrd memory
 * @details The vector is templated on the stored type and size.
 * - Allocation: pre-allocate sizeof(SparseVector) and call the static allocator:
 *   SparseVector<Type,N>::MakeInstance(void *addr)
 * - Use the base class SparseVectorInterface<Type> as main interface. It provides
 *   iterators and random access,
 * - Distributing concurrently vector elements, withing the available size:
 *     Type *elem = vect->next_free(parameters);
 *   The compiler will match the parameters with those of a constructor of Type
 * - Releasing elements at a given index:
 *     vect->release(index);
 * - Getting number of shared and used elements:
 *     SparseVectorInterface::size() / size_used();
 * - Sparsity is defined as used/shared, shared fraction is nshared/capacity
 * - The SparseVectorInterface provides static methods for predicate-based selection of elements:
 *     select(SparseVectorInterface<Type> *vect, Predicate select_func, Container *ind, unsigned *nsel)
 *       vect - Vector to select from
 *       select_func - predicate function having the format:
 *         __device__ bool func(int, const Vector_t*) // index in the vector and const vector pointer
 *       ind - containter holding the indices of the selected elements
 *       nsel - number of selected elements
 *     select_and_move(SparseVectorInterface<Type> *source, Predicate select_func,
 *                     SparseVectorInterface<Type> *dest, unsigned nsel)
 *       source - source vector
 *       select_func - selection predicate function
 *       dest - destination vector
 *       nsel - number of selected elements
 *       The selected elements are removed from source and added at the end of dest.
 *     relese_selected(SparseVectorInterface<Type> *source, Container *ind, unsigned *nsel)
 *       Releases nsel elements stored in the ind vector
 *     select_used(SparseVectorInterface<Type> *source, Container *ind, unsigned *nsel)
 *       Copies the indices of all used elements in ind.
 *     compact(SparseVectorInterface<Type> *source, SparseVectorInterface<Type> *source, unsigned *nsel)
 *       Moves all used elements from source to the end of dest, returning the number of copied elements nsel
 *
 * @author Andrei Gheata (andrei.gheata@cern.ch)
 */

#ifndef ADEPT_SPARSEVECTOR_H_
#define ADEPT_SPARSEVECTOR_H_

#include <CopCore/CopCore.h>
#include <AdePT/Atomic.h>

namespace adept {

template <typename Type>
class SparseVectorInterface;

template <typename Type, unsigned N>
class SparseVector;

namespace sa_detail {

#ifdef COPCORE_CUDA_COMPILER

__device__ inline int lane_id(void)
{
  return threadIdx.x & 31;
}

template <typename Type>
__device__ inline bool is_used(int index, const SparseVectorInterface<Type> *svector)
{
  return svector->is_used(index);
}

__device__ void print_mask(unsigned int mask)
{
  for (int lane = 0; lane < 32; ++lane) {
    if ((mask & (1 << lane)))
      printf("1");
    else
      printf("0");
  }
  printf("\n");
}

template <typename Type, unsigned N>
__global__ void construct_vector(void *addr, int numSMs)
{
  auto vect = new (addr) SparseVector<Type, N>(N);
  vect->setNumSMs(numSMs);
}

template <typename Type>
__global__ void release_selected_kernel(SparseVectorInterface<Type> *svect, const unsigned int *selection,
                                        unsigned *nselected)
{
  // Release the selected entries
  for (auto tid = blockIdx.x * blockDim.x + threadIdx.x; tid < *nselected; tid += blockDim.x * gridDim.x)
    svect->release(selection[tid]);
}

template <typename Type>
__global__ void release_selected_kernel_launcher(SparseVectorInterface<Type> *svect, const unsigned int *selection,
                                                 unsigned *nselected)
{
  // Launcher for releasing the selected entries
  constexpr unsigned int warpsPerSM = 32; // target occupancy
  constexpr unsigned int block_size = 256;
  unsigned int grid_size =
      min(warpsPerSM * svect->getNumSMs() * 32 / block_size, (*nselected + block_size - 1) / block_size);
  // printf("running release_selected_kernel<<<%d, %d>>>\n", grid_size, block_size);
  sa_detail::release_selected_kernel<Type><<<grid_size, block_size>>>(svect, selection, nselected);
}

template <typename Type>
__global__ void clear_kernel(SparseVectorInterface<Type> *svect)
{
  // Clear the vector
  svect->clear();
}

/// Make a selection of elements according to the user predicate and write it to the output.
/// Assume that IndexContainer::operator[] is implemented and use it to copy selected element indices
template <typename Type, typename Predicate, typename IndexContainer>
__global__ void select_kernel(const SparseVectorInterface<Type> *svect, Predicate pred_func, IndexContainer *output,
                              unsigned *nselected)

{
  // Non-order-preserving stream compacting algorithm
  // Based on algorithm described in https://www.jstage.jst.go.jp/article/ijnc/7/2/7_208/_pdf/-char/ja
  // nselected content must be zeroed before calling the kernel
  constexpr unsigned int warp_mask = 0xFFFFFFFF;

  int num_items = svect->size();
  int tid       = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id   = tid / 32;
  // exit if the current warp is out of range
  if (warp_id > num_items / 1024) return;
  int lnid          = lane_id();
  unsigned votes    = 0;
  unsigned predmask = 0;
  int cnt           = 0;
  // Stage 1: each warp iterates 32 subgroups of 32 elements each. Each thread computes the selection
  // predicates for one element in the group and combine them into a `votes` word using __ballot. The number of
  // votes in subgroup `i` is stored into the variable cnt{i} (value for lane `i` of the warp). The votes
  // for subgroup `i` are stored in predmask{i}
  int group_max = min(32, 1 + (num_items - 1024 * warp_id) / 32);
  for (int group_id = 0; group_id < group_max; ++group_id) {
    // Get votes for this group
    int index     = 1024 * warp_id + 32 * group_id + lnid;
    bool selected = (index < num_items) ? is_used(index, svect) && pred_func(index, svect) : false;

    votes = __ballot_sync(warp_mask, selected);

    // store the ballot and the votes count in the lane matching the group id
    if (lnid == group_id) {
      predmask = votes;
      cnt      = __popc(votes); // number of votes (set bits) in the subgroup
      // printf("(warp_id=%d, group_id=%d) cnt = %d  predmask = 0x%x ", warp_id, lnid, cnt, predmask);
      // print_mask(predmask);
    }
  }

// Stage 2: parallel prefix sum of all cnt variables giving the offset for each subgroup
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    int n = __shfl_up_sync(warp_mask, cnt, i);
    if (lnid >= i) cnt += n;
  }

  // cnt for lane 31 now contains the sum of all votes in all ballots for warp_id

  // Stage 3: the last lane thread adds the sum of all votes in the warp to get all group offsets.
  // Each group offset is broadcasted to all threads in the warp using __shfl_sync
  int group_offset = 0;
  if (lnid == 31) {
    // printf("final count for warp %d: cnt = %d\n", warp_id, cnt);
    group_offset = atomicAdd(nselected, cnt);
    // printf("warp %d: group_offset = %d  all_votes = %d\n", warp_id, group_offset, *nselected);
  }
  group_offset = __shfl_sync(warp_mask, group_offset, 31);

  // Stage 4: Write the indices of the selected elements to the output vector
  for (int group_id = 0; group_id < group_max; ++group_id) {
    int mask           = __shfl_sync(warp_mask, predmask, group_id); // broadcast mask stored by lane group_id
    int subgroup_index = 0;
    if (group_id > 0)
      subgroup_index = __shfl_sync(warp_mask, cnt, group_id - 1); // broadcast from thr group_id - 1 if group_id > 0

    if (mask & (1 << lnid)) { // each thr extracts its pred bit
      int idest     = group_offset + subgroup_index + __popc(mask & ((1 << lnid) - 1));
      int isrc      = 1024 * warp_id + 32 * group_id + lnid;
      output[idest] = isrc;
    }
  }
}

/// Dynamic launcher for select_kernel
template <typename Type, typename Predicate, typename IndexContainer>
__global__ void select_kernel_launcher(const SparseVectorInterface<Type> *svect, Predicate pred_func,
                                       IndexContainer *output, unsigned *nselected)
{
  const int num_threads = 128;
  int num_items         = svect->size();
  int num_blocks        = (num_items + 4095) / 4096; // a warp processes groups of 1024 elements
  *nselected            = 0;
  sa_detail::select_kernel<Type, Predicate, IndexContainer>
      <<<num_blocks, num_threads>>>(svect, pred_func, output, nselected);
}

/// Make a selection of elements according to the user predicate and write it to the output.
/// Copy directly elements at the end of the output vector.
/// Similar to select_kernel, but having extra actions applied to the vectors
template <typename Type, typename Predicate>
__global__ void select_and_move_kernel(SparseVectorInterface<Type> *svect, Predicate pred_func,
                                       SparseVectorInterface<Type> *output, unsigned *nselected, int dest_offset)

{
  // Non-order-preserving stream compacting algorithm
  // Based on algorithm described in https://www.jstage.jst.go.jp/article/ijnc/7/2/7_208/_pdf/-char/ja
  // nselected content must be zeroed before calling the kernel
  constexpr unsigned int warp_mask = 0xFFFFFFFF;

  assert(output != svect && "Compacting into the same container not supported yet");
  int num_items = svect->size();
  int tid       = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id   = tid / 32;
  // exit if the current warp is out of range
  if (warp_id > num_items / 1024) return;
  int lnid          = lane_id();
  unsigned votes    = 0;
  unsigned predmask = 0;
  int cnt           = 0;
  // Stage 1: each warp iterates 32 subgroups of 32 elements each. Each thread computes the selection
  // predicates for one element in the group and combine them into a `votes` word using __ballot. The number of
  // votes in subgroup `i` is stored into the variable cnt{i} (value for lane `i` of the warp). The votes
  // for subgroup `i` are stored in predmask{i}
  int group_max = min(32, 1 + (num_items - 1024 * warp_id) / 32);
  for (int group_id = 0; group_id < group_max; ++group_id) {
    // Get votes for this group
    int index     = 1024 * warp_id + 32 * group_id + lnid;
    bool selected = (index < num_items) ? is_used(index, svect) && pred_func(index, svect) : false;

    votes = __ballot_sync(warp_mask, selected);

    // store the ballot and the votes count in the lane matching the group id
    if (lnid == group_id) {
      predmask = votes;
      cnt      = __popc(votes); // number of votes (set bits) in the subgroup
      //  mask out the selected elements that will be released from the initial vector
      int indmask      = index / 32;
      unsigned newmask = svect->mask_at(indmask).load() & (~predmask);
      svect->mask_at(indmask).store(newmask);
      // printf("(warp_id=%d, group_id=%d) cnt = %d  predmask = 0x%x ", warp_id, lnid, cnt, predmask);
      // print_mask(predmask);
    }
  }

// Stage 2: parallel prefix sum of all cnt variables giving the offset for each subgroup
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    int n = __shfl_up_sync(warp_mask, cnt, i);
    if (lnid >= i) cnt += n;
  }

  // cnt for lane 31 now contains the sum of all votes in all ballots for warp_id

  // Stage 3: the last lane thread adds the sum of all votes in the warp to get all group offsets.
  // Each group offset is broadcasted to all threads in the warp using __shfl_sync
  int group_offset = 0;
  if (lnid == 31) {
    // printf("final count for warp %d: cnt = %d\n", warp_id, cnt);
    group_offset = atomicAdd(nselected, cnt);
    output->add_elements(cnt);   // increase the counter in the output vector and set masks
    svect->remove_elements(cnt); // remove used elements from the source
    // printf("warp %d: group_offset = %d  all_votes = %d\n", warp_id, group_offset, *nselected);
  }
  group_offset = __shfl_sync(warp_mask, group_offset, 31);

  // Stage 4: Write the indices of the selected elements to the output vect
  for (int group_id = 0; group_id < group_max; ++group_id) {
    int mask           = __shfl_sync(warp_mask, predmask, group_id); // broadcast mask stored by lane group_id
    int subgroup_index = 0;
    if (group_id > 0)
      subgroup_index = __shfl_sync(warp_mask, cnt, group_id - 1); // broadcast from thr group_id - 1 if group_id > 0

    if (mask & (1 << lnid)) { // each thr extracts its pred bit
      int idest = dest_offset + group_offset + subgroup_index + __popc(mask & ((1 << lnid) - 1));
      int isrc  = 1024 * warp_id + 32 * group_id + lnid;
      // printf("== copy from %d to %d\n", isrc, idest);
      new (output->data() + idest) Type((*svect)[isrc]); // call in-place copy constructor starting with last element
    }
  }
}

/// Dynamic launcher for select_kernel
template <typename Type, typename Predicate>
__global__ void select_and_move_kernel_launcher(SparseVectorInterface<Type> *svect, Predicate pred_func,
                                                SparseVectorInterface<Type> *output, unsigned *nselected)
{
  using Vector_t            = SparseVectorInterface<Type>;
  const int num_items       = svect->size();
  constexpr int num_threads = 128;
  const int num_blocks      = (num_items + 4095) / 4096;
  *nselected                = 0;
  sa_detail::select_and_move_kernel<Type>
      <<<num_blocks, num_threads>>>(svect, pred_func, output, nselected, output->size());
}

#endif // COPCORE_CUDA_COMPILER
} // end namespace sa_detail

/** @brief SparseVector interface */
template <typename Type>
class SparseVectorInterface {
  using Vector_t               = SparseVectorInterface<Type>;
  using value_type             = Type;
  using pointer                = value_type *;
  using const_pointer          = const value_type *;
  using reference              = value_type &;
  using const_reference        = const value_type &;
  using iterator               = value_type *;
  using const_iterator         = const value_type *;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_t                 = std::size_t;
  using Index_t                = unsigned int;
  using Mask_t                 = unsigned int;
  using AtomicInt_t            = adept::Atomic_t<int>;
  using AtomicMask_t           = adept::Atomic_t<Mask_t>;
  using LaunchGrid_t           = copcore::launch_grid<copcore::BackendType::CUDA>;

protected:
  void *fBegin     = nullptr; ///< Start address of the vector data
  void *fMaskBegin = nullptr; ///< Start address of the mask data
  AtomicInt_t fNbooked;       ///< Number of booked elements (requested concurrently, may become larger than N)
  AtomicInt_t fNshared;       ///< Number of shared elements (given away, decreases only after clear/compact)
  AtomicInt_t fNused;         ///< Number of elements currently in use
  unsigned fCapacity{0};      ///< Capacity of the vector
  unsigned fMaskCapacity{0};  ///< Capacity of the mask vector
  int fNumSMs{0};             ///< number of streaming multi-processors

  __host__ __device__ SparseVectorInterface() = delete;

  __host__ __device__ SparseVectorInterface(size_t totalCapacity)
      : fCapacity(totalCapacity), fMaskCapacity(totalCapacity / 32)
  {
  }

  __host__ __device__ SparseVectorInterface(const SparseVectorInterface &other) = delete;

  __host__ __device__ const SparseVectorInterface &operator=(const SparseVectorInterface &other) = delete;

  __host__ __device__ void setNumSMs(int numSMs) { fNumSMs = numSMs; }

  /** @brief Set the bit mask for element i */
  __host__ __device__ __forceinline__ void set_bit(Index_t i)
  {
    Index_t indmask = i / 32;
    Mask_t contrib  = 1 << (i & 31);
    // reads and try to update existing mask
    Mask_t mask = mask_at(indmask).load();
    while (!mask_at(indmask).compare_exchange_strong(mask, mask | contrib))
      ;
  }

  /** @brief Reset the bit mask for element i */
  __host__ __device__ __forceinline__ void reset_bit(Index_t i)
  {
    Index_t indmask = i / 32;
    Mask_t contrib  = ~(1 << (i & 31));
    // reads and try to update existing mask
    Mask_t mask = mask_at(indmask).load();
    while (!mask_at(indmask).compare_exchange_strong(mask, mask & contrib))
      ;
  }

public:
  /** @brief Maximum number of elements */
  __host__ __device__ __forceinline__ constexpr size_t capacity() const { return fCapacity; }

  /** @brief Number of shared elements (watermark for elements given away) */
  __host__ __device__ __forceinline__ size_t size() const { return fNshared.load(); }

  /** @brief Number of elements still in use */
  __host__ __device__ __forceinline__ size_t size_used() const { return fNused.load(); }

  /** @brief Number of elements booked. Can only exceed the shared size when the vector is full */
  __host__ __device__ __forceinline__ size_t size_booked() const { return fNbooked.load(); }

  /** @brief Is container empty */
  __host__ __device__ __forceinline__ bool empty() const { return !size(); }

  /** @brief Get number of SMs on the current card (TODO: to be moved to common copcoreutils) */
  __host__ __device__ int getNumSMs() const { return fNumSMs; }

  /// Forward iterator methods. Note: it iterates through holes as well.
  __host__ __device__ __forceinline__ iterator begin() { return (iterator)this->fBegin; }
  __host__ __device__ __forceinline__ const_iterator begin() const { return (const_iterator)this->fBegin; }
  __host__ __device__ __forceinline__ iterator end() { return begin() + size(); }
  __host__ __device__ __forceinline__ const_iterator end() const { return begin() + size(); }

  /// Backward iterator methods. Note: it iterates through holes as well.
  __host__ __device__ __forceinline__ iterator rbegin() { return (iterator)this->fBegin; }
  __host__ __device__ __forceinline__ const_iterator rbegin() const { return (const_iterator)this->fBegin; }
  __host__ __device__ __forceinline__ iterator rend() { return begin() + size(); }
  __host__ __device__ __forceinline__ const_iterator rend() const { return begin() + size(); }

  /// Mask vector accessors
  __host__ __device__ __forceinline__ AtomicMask_t *mask_begin() { return (AtomicMask_t *)this->fMaskBegin; }
  __host__ __device__ __forceinline__ const AtomicMask_t *mask_begin() const
  {
    return (const AtomicMask_t *)this->fMaskBegin;
  }
  __host__ __device__ __forceinline__ AtomicMask_t &mask_at(size_t i) { return mask_begin()[i]; }
  __host__ __device__ __forceinline__ const AtomicMask_t &mask_at(size_t i) const { return mask_begin()[i]; }

  /// Return a pointer to the vector's buffer, even if empty().
  __host__ __device__ __forceinline__ pointer data() { return pointer(begin()); }
  /// Return a pointer to the vector's buffer, even if empty().
  __host__ __device__ __forceinline__ const_pointer data() const { return const_pointer(begin()); }

  /// Element accessors
  __host__ __device__ __forceinline__ reference operator[](size_t idx)
  {
    assert(idx < size());
    return begin()[idx];
  }

  __host__ __device__ __forceinline__ const_reference operator[](size_t idx) const
  {
    assert(idx < size());
    return begin()[idx];
  }

  __host__ __device__ __forceinline__ reference front()
  {
    assert(!empty());
    return begin()[0];
  }

  __host__ __device__ __forceinline__ const_reference front() const
  {
    assert(!empty());
    return begin()[0];
  }

  __host__ __device__ __forceinline__ reference back()
  {
    assert(!empty());
    return end()[-1];
  }

  __host__ __device__ __forceinline__ const_reference back() const
  {
    assert(!empty());
    return end()[-1];
  }

  /** @brief Sparsity defined as 1 - nused/nshared (0 means no hole)*/
  __host__ __device__ __forceinline__ float get_sparsity() const
  {
    return (size()) ? 1. - (float)size_used() / size() : 0.;
  }

  /** @brief Shared fraction defined as nshared/nmax (0 means empty, 1 means full)*/
  __host__ __device__ __forceinline__ float get_shared_fraction() const { return (float)size() / capacity(); }

  /** @brief Selected fraction defined as nselected/nused*/
  __host__ __device__ __forceinline__ float get_selected_fraction(size_t nselected) const
  {
    return (size_used()) ? (float)nselected / size_used() : 0.;
  }

  /** @brief Clear the content */
  __host__ __device__ __forceinline__ void clear()
  {
    fNused.store(0);
    fNshared.store(0);
    fNbooked.store(0);
    // memset(mask_begin(), 0, fMaskCapacity * sizeof(Mask_t)); // do we need to clear the storage as well?
  }

  /** @brief Add elements at the end of the vector */
  __host__ __device__ void add_elements(unsigned n)
  {
    // this should be protected, but has to be launched as a templated kernel...
    // update the mask from bits fNshared to fNshared + n in one go
#ifndef COPCORE_CUDA_COMPILER
    using std::min;
#endif
    constexpr unsigned full32 = 0xFFFFFFFFu;
    auto set_bits             = [](unsigned from, unsigned to) {
      return (to - from == 31) ? full32 : (~(full32 << (to + 1 - from))) << from;
    };
    unsigned start     = fNshared.fetch_add(n);
    unsigned remaining = n;
    unsigned indmask   = start / 32;
    unsigned start_bit = start & 31;
    while (remaining > 0) {
      unsigned end_bit = min(start_bit + remaining - 1, 31u);
      Mask_t contrib   = set_bits(start_bit, end_bit);
      if (contrib == full32)
        mask_at(indmask).store(contrib);
      else {
        // reads and try to update existing mask
        Mask_t mask = mask_at(indmask).load();
        while (!mask_at(indmask).compare_exchange_strong(mask, mask | contrib))
          ;
      }
      remaining -= end_bit - start_bit + 1;
      indmask++;
      start_bit = 0;
    }

    fNused += n;
    fNbooked += n;
  }

  /** @brief Update counters when removing elements */
  __host__ __device__ __forceinline__ void remove_elements(unsigned n) { fNused -= n; }

  /** @brief Dispatch next free element at the end, nullptr if none left. Construct in place using provided params */
  template <typename... T>
  __host__ __device__ __forceinline__ pointer next_free(const T &... params)
  {
    // Operation may fail if the max size is exceeded. Has to be checked by the user.
    int index = fNbooked.fetch_add(1);
    if (index >= fCapacity) return nullptr;
    // construct in place
    new ((data() + index)) Type(params...);
    fNshared++;
    fNused++;

    // update the mask - this may become blocking, we can reduce contention using warp-level synchronization
    // to have a single thread updating the atomic:
    //  - ballot the requests coming from threads of the same warp
    //  - count the requests and let one thread do the atomic increment and get the base offset before incrementing
    //  - broadcast the offet to all threads in the warp
    //  - compute the index to be assigned to each request
    // TODO: use the `Opportunistic Warp-level Programming` approach from:
    //  https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    set_bit(index);
    return (data() + index);
  }

  /** @brief Release the element at index i */
  __host__ __device__ __forceinline__ void release(Index_t i)
  {
    // Make sure the element is not already released
    assert(is_used(i) && "Trying to release an element not in use.");
    fNused--;
    reset_bit(i);
  }

  /** @brief Check if the element with index i is used */
  __host__ __device__ __forceinline__ bool is_used(Index_t i) const
  {
    assert(i < fCapacity);
    return (mask_at(i / 32).load() & (1 << (i & 31)));
  }

  /** @brief Check if container is fully distributed */
  __host__ __device__ __forceinline__ bool is_full() const { return (size() == fCapacity); }

  /** @brief Check if container is empty */
  __host__ __device__ __forceinline__ bool is_empty() const { return (size_used() == 0); }

  /** @brief Fills selection vector fSelected with element indices passing the user predicate
   *  @param pred Predicate function taking as arguments the SparseVector pointer and the element index
   *  @returns Number of selected elements. Must be reset to 0 when calling the function.
   */
#ifdef COPCORE_CUDA_COMPILER
  template <typename Predicate, typename Container>
  static void select(const Vector_t *svect, Predicate pred_func, Container *output, unsigned *num_selected)
  {
    sa_detail::select_kernel_launcher<Type, Predicate, Container>
        <<<1, 1>>>((Vector_t *)svect, pred_func, output, num_selected);
  }
#else
  template <typename Predicate, typename Container>
  static void select(const Vector_t *svect, Predicate pred_func, Container *output, unsigned *num_selected)
  {
    // host version
    const int num_items = svect->size();
    *num_selected       = 0;
    for (int i = 0; i < num_items; ++i) {
      if (svect->is_used(i) && pred_func(i, svect)) output[(*num_selected)++] = i;
    }
  }
#endif

  /** @brief Fills selection vector fSelected with element indices passing the user predicate
   *  @param pred Predicate function taking as arguments the SparseVector pointer and the element index
   *  @returns Number of selected elements. Must be reset to 0 when calling the function.
   */
#ifdef COPCORE_CUDA_COMPILER
  template <typename Predicate>
  static void select_and_move(Vector_t *svect, Predicate pred_func, Vector_t *output, unsigned *num_selected)
  {
    sa_detail::select_and_move_kernel_launcher<Type><<<1, 1>>>(svect, pred_func, output, num_selected);
  }
#else
  template <typename Predicate>
  static void select_and_move(Vector_t *svect, Predicate pred_func, Vector_t *output, unsigned *num_selected)
  {
    // host version
    // copy to destination vect
    int mask_size         = svect->size() / 32;
    const int dest_offset = output->size();
    *num_selected         = 0;
    for (auto im = 0; im <= mask_size; ++im) {
      Mask_t mask = svect->mask_at(im).load();
      for (auto i = 0; i < 32; ++i) {
        if ((mask & (1 << (i & 31))) && pred_func(32 * im + i, svect)) {
          int idest = dest_offset + (*num_selected)++;
          new (output->data() + idest) Type((*svect)[32 * im + i]);
          output->set_bit(idest);
          mask &= ~(1 << (i & 31)); // remove selected element from mask
          svect->mask_at(im).store(mask);
        }
      }
    }
    output->fNused += *num_selected;
    output->fNshared += *num_selected;
    svect->fNused -= *num_selected;
  }
#endif

  template <typename Container>
  static void select_used(const Vector_t *svect, Container *output, unsigned *num_selected)
  {
    Vector_t::select(svect, [] __device__(int, const Vector_t *) { return true; }, output, num_selected);
  }

  /** @brief Compacts used elements of this vect into a destination vect. */
#ifdef COPCORE_CUDA_COMPILER
  static void compact(Vector_t *svect, Vector_t *output, unsigned *num_selected)
  {
    // copy to destination vector
    sa_detail::select_and_move_kernel_launcher<Type>
        <<<1, 1>>>(svect, [] __device__(int, const Vector_t *) { return true; }, output, num_selected);
    // update output vector
    sa_detail::clear_kernel<Type><<<1, 1>>>(svect);
  }
#else
  static void compact(Vector_t *svect, Vector_t *output, unsigned *num_selected)
  {
    // host version
    // copy to destination vector
    int mask_size         = svect->size() / 32;
    const int dest_offset = output->size();
    *num_selected         = 0;
    for (auto im = 0; im <= mask_size; ++im) {
      Mask_t mask = svect->mask_at(im).load();
      for (auto i = 0; i < 32; ++i) {
        if (mask & (1 << (i & 31))) new (output->data() + dest_offset + (*num_selected)++) Type((*svect)[32 * im + i]);
      }
    }
    output->fNused += *num_selected;
    output->fNshared += *num_selected;
    svect->clear();
  }
#endif

  /** @brief Fills selection vector fSelected with indices of remaining used elements. */
#ifdef COPCORE_CUDA_COMPILER
  static void release_selected(Vector_t *svect, unsigned *selection, unsigned *nselected)
  {
    // we pass n_elements by pointer to avoid having to copy the value to the host
    sa_detail::release_selected_kernel_launcher<Type><<<1, 1>>>(svect, selection, nselected);
  }
#else
  static void release_selected(Vector_t *svect, unsigned *selection, unsigned *nselected)
  {
    // host version
    for (int i = 0; i < *nselected; ++i)
      svect->release(selection[i]);
  }
#endif
};

/** @brief A (non-resizeable variable size vect adopting memory and having elements added in an atomic way */
template <typename Type, unsigned N = 1 << 16>
class SparseVector : public SparseVectorInterface<Type> {
  using Base_t = SparseVectorInterface<Type>;
  static_assert(N > 0 && 32 * (N >> 5) == N, "adept::SparseVector capacity must be multiple of 32");
  static_assert(std::is_copy_constructible<Type>::value,
                "adept::SparseVector Error: stored type is not copy constructible");

private:
  alignas(Type) char fData[N * sizeof(Type)]; // array of real data (correct alignment, but avoid calling the default
                                              // constructor)
  unsigned int fMasks[N / 32];                // array of masks

#ifdef COPCORE_CUDA_COMPILER
  friend void sa_detail::construct_vector<Type, N>(void *, int);
#endif
  __host__ __device__ SparseVector(size_t totalCapacity) : SparseVectorInterface<Type>(totalCapacity)
  {
    Base_t::fBegin     = fData;
    Base_t::fMaskBegin = fMasks;
  }

  __host__ __device__ SparseVector(const SparseVector &other) = delete;

  __host__ __device__ const SparseVector &operator=(const SparseVector &other) = delete;

protected:
  static SparseVector *ConstructOnDevice(void *addr)
  {
#ifndef COPCORE_CUDA_COMPILER
    return nullptr; // should never happen
#else
    int numSMs = copcore::get_num_SMs();
    sa_detail::construct_vector<Type, N><<<1, 1>>>(addr, numSMs);
    return reinterpret_cast<SparseVector<Type, N> *>(addr);
#endif
  }

public:
  __host__ static SparseVector *MakeInstanceAt(void *addr = nullptr)
  {
    if (!addr) return new SparseVector(N);
    bool devicePtr = copcore::is_device_pointer(addr);
    if (devicePtr) return SparseVector::ConstructOnDevice(addr);
    return new (addr) SparseVector(N);
  }

}; // End class SparseVector
} // End namespace adept

#endif // ADEPT_SPARSEVECTOR_H_
