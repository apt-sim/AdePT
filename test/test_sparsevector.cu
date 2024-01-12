// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <CopCore/Ranluxpp.h>
#include <AdePT/base/SparseVector.h>

#include <VecGeom/base/Stopwatch.h>

/** The test fills a sparse vector with tracks having random energy. It demonstrates allocation,
concurrent distribution of elements, selection based on a lambda predicate function, gathering
of used slots in a selection vector, compacting elements by copy-constructing in a second sparse vector.
 */

/// A simple track
struct Track_t {
  using Rng_t = RanluxppDouble;

  Rng_t rng;
  float energy{0.};
  bool alive{true};

  // a default constructor is not necessarily needed
  // constructor parameters (or copy constructor) can be passed via SparseVectorImplementation::next_free()
  __host__ __device__ Track_t(unsigned itr)
  {
    rng.SetSeed(itr);
    energy = (float)rng.Rndm();
  }
};

// some utility kernels for filling the vector concurrently and printing info (vector resides on device)

__global__ void fill_tracks(adept::SparseVectorInterface<Track_t> *vect1_ptr, int num_elem)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_elem) return;
  // parameters of next_free are passed to the matching constructor called in place
  Track_t *track = vect1_ptr->next_free(tid);
  if (!track) COPCORE_EXCEPTION("Out of vector space");
}

__global__ void print_tracks(adept::SparseVectorInterface<Track_t> *tracks, int start, int num)
{
  const int nshared = tracks->size();
  printf(" data: ");
  for (int i = start; i < start + num && i < nshared; ++i) {
    printf(" %.2f", (*tracks)[i].energy);
    if (!tracks->is_used(i)) printf("x");
  }
  printf("...\n");
}

__global__ void print_selected_tracks(adept::SparseVectorInterface<Track_t> *tracks, const unsigned *selection,
                                      const unsigned *n_selected, int start, int num)
{
  printf("selected %d tracks:\n > ", *n_selected);
  int limit = min(*n_selected, start + num);
  for (int i = start; i < limit; ++i) {
    printf("%.2f ", (*tracks)[selection[i]].energy);
  }
  printf("...\n");
}

__global__ void reset_selection(unsigned *nselected)
{
  *nselected = 0;
}

template <typename Vector_t>
__global__ void print_vector(int iarr, Vector_t *vect)
{
  printf("=== vect %d: fNshared=%lu/%lu fNused=%lu fNbooked=%lu - shared=%.1f%% sparsity=%.1f%%\n", iarr, vect->size(),
         vect->capacity(), vect->size_used(), vect->size_booked(), 100. * vect->get_shared_fraction(),
         100. * vect->get_sparsity());
}

template <typename Vector_t, typename Function>
__global__ void get_vector_data(const Vector_t *vect, Function vect_func, int *data)
{
  // data should be allocated in managed memory, vect_func should call a getter of Vector_t
  *data = vect_func(vect);
}

/// Test performance-critical SparseVector operations, executing as kernels. The syncronization
/// operations exposed are only for timing purposes, the operations are valid also without.
//____________________________________________________________________________________________________
int main(void)
{
  constexpr int VectorSize = 1 << 20;
  int ntracks              = 1000000;
  using Vector_t = adept::SparseVector<Track_t, VectorSize>; // 1<<16 is the default vector size if parameter omitted
  using VectorInterface = adept::SparseVectorInterface<Track_t>;

  vecgeom::Stopwatch timer;

  Vector_t *vect1_ptr_d, *vect2_ptr_d;
  unsigned *sel_vector_d;
  unsigned *nselected_hd;
  printf("Running on %d tracks. Size of adept::SparseVector<Track_t, %d> = %lu\n", ntracks, VectorSize,
         sizeof(Vector_t));
  // allocation can be done on device or managed memory
  COPCORE_CUDA_CHECK(cudaMalloc(&vect1_ptr_d, sizeof(Vector_t)));
  COPCORE_CUDA_CHECK(cudaMalloc(&vect2_ptr_d, sizeof(Vector_t)));
  COPCORE_CUDA_CHECK(cudaMalloc(&sel_vector_d, VectorSize * sizeof(unsigned)));
  COPCORE_CUDA_CHECK(cudaMallocManaged(&nselected_hd, sizeof(unsigned)));

  // managed variables to read state from device
  int *nshared, *nused, *nselected;
  COPCORE_CUDA_CHECK(cudaMallocManaged(&nshared, 2 * sizeof(int)));
  COPCORE_CUDA_CHECK(cudaMallocManaged(&nused, 2 * sizeof(int)));
  COPCORE_CUDA_CHECK(cudaMallocManaged(&nselected, 2 * sizeof(int)));

  // static allocator for convenience
  Vector_t::MakeInstanceAt<copcore::BackendType::CUDA>(vect1_ptr_d);
  Vector_t::MakeInstanceAt<copcore::BackendType::CUDA>(vect2_ptr_d);

  reset_selection<<<1, 1>>>(nselected_hd);

  // Construct and distribute tracks concurrently
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  timer.Start();
  fill_tracks<<<(ntracks + 127) / 128, 128>>>(vect1_ptr_d, ntracks);
  get_vector_data<<<1, 1>>>(vect1_ptr_d, [] __device__(const VectorInterface *arr) { return arr->size(); }, nshared);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_fill = timer.Stop();
  std::cout << "time_construct_and_share = " << time_fill << std::endl;
  print_vector<<<1, 1>>>(1, vect1_ptr_d);
  print_tracks<<<1, 1>>>(vect1_ptr_d, 0, 32); // print just first 32 tracks
  int nfilled = *nshared;
  if (nfilled != ntracks) {
    std::cerr << "Error in next_free.\n";
    return 1;
  }

  // Select tracks with energy < 0.2
  // *** note that we can use any device predicate function with the prototype:
  //   __device__ bool func(int, const Vector_t*) // index in the vector and const vector pointer
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  timer.Start();
  auto select_func = [] __device__(int i, const VectorInterface *arr) { return ((*arr)[i].energy < 0.2); };
  VectorInterface::select(vect1_ptr_d, select_func, sel_vector_d, nselected_hd);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_select = timer.Stop();
  int nselected1   = *nselected_hd;
  std::cout << "\ntime_select for " << nselected1 << " tracks with (energy < 0.2) = " << time_select << std::endl;
  print_vector<<<1, 1>>>(1, vect1_ptr_d);
  print_selected_tracks<<<1, 1>>>(vect1_ptr_d, sel_vector_d, nselected_hd, 0, 32);
  if (nselected1 == 0) {
    std::cerr << "Error in select: 0 tracks.\n";
    return 2;
  }

  // Release the tracks we just selected, creating holes in the vector
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  timer.Start();
  VectorInterface::release_selected(vect1_ptr_d, sel_vector_d, nselected_hd);
  get_vector_data<<<1, 1>>>(vect1_ptr_d, [] __device__(const VectorInterface *arr) { return arr->size_used(); }, nused);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_release = timer.Stop();
  std::cout << "\ntime_release_selected = " << time_release << "   nused = " << *nused << std::endl;
  print_vector<<<1, 1>>>(1, vect1_ptr_d);
  print_tracks<<<1, 1>>>(vect1_ptr_d, 0, 32);
  int nused_after_release = *nused;
  if ((nselected1 + nused_after_release) != ntracks) {
    std::cerr << "Error in release_selected.\n";
    return 3;
  }

  // Demonstrate select_and_move functionality
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  timer.Start();
  // a fuction selecting tracks having energy > 0.8. We move these tracks in a second vector
  auto select2_func = [] __device__(int i, const VectorInterface *arr) { return ((*arr)[i].energy > 0.8); };
  //===
  VectorInterface::select_and_move(vect1_ptr_d, select2_func, vect2_ptr_d, nselected_hd);
  //===
  auto time_select_and_move = timer.Stop();
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  get_vector_data<<<1, 1>>>(vect1_ptr_d, [] __device__(const VectorInterface *arr) { return arr->size_used(); },
                            &nused[0]);
  get_vector_data<<<1, 1>>>(vect2_ptr_d, [] __device__(const VectorInterface *arr) { return arr->size_used(); },
                            &nused[1]);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "\ntime_select_and_move (energy > 0.8) = " << time_select_and_move << std::endl;
  print_vector<<<1, 1>>>(1, vect1_ptr_d);
  print_tracks<<<1, 1>>>(vect1_ptr_d, 0, 32);
  print_vector<<<1, 1>>>(2, vect2_ptr_d);
  print_tracks<<<1, 1>>>(vect2_ptr_d, 0, 32);
  // Check the moved tracks
  int nused_after_move  = nused[0];
  int nused_after_move2 = nused[1];
  if ((nused_after_release - nused_after_move) != nused_after_move2) {
    std::cerr << "Error in select_and_move.\n";
    return 4;
  }

  // Demonstrate a common selection method that should be used when the vector is fragmented.
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  timer.Start();
  VectorInterface::select_used(vect1_ptr_d, sel_vector_d, nselected_hd);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_select_used = timer.Stop();
  std::cout << "\ntime_select_used = " << time_select_used << std::endl;
  print_selected_tracks<<<1, 1>>>(vect1_ptr_d, sel_vector_d, nselected_hd, 0, 32);
  if (*nselected_hd != nused_after_move) {
    std::cerr << "Error in select_used.\n";
    return 5;
  }

  // Compact used elements by copying them into a destination vector. The stage above should be preferred
  // if the sparsity is small, while this one is preffered for high sparsity. See SparseVector header
  // for the definition of sparsity, shared and selected fractions.
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  timer.Start();
  VectorInterface::compact(vect1_ptr_d, vect2_ptr_d, nselected_hd);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  auto time_compact = timer.Stop();
  get_vector_data<<<1, 1>>>(vect1_ptr_d, [] __device__(const VectorInterface *arr) { return arr->size_used(); },
                            &nused[0]);
  get_vector_data<<<1, 1>>>(vect2_ptr_d, [] __device__(const VectorInterface *arr) { return arr->size_used(); },
                            &nused[1]);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "\ntime_compact = " << time_compact << std::endl;
  print_vector<<<1, 1>>>(1, vect1_ptr_d);
  print_vector<<<1, 1>>>(2, vect2_ptr_d);
  print_tracks<<<1, 1>>>(vect2_ptr_d, 0, 32);
  COPCORE_CUDA_CHECK(cudaDeviceSynchronize());
  if ((nused[0] != 0) || (nused[1] != nused_after_move2 + nused_after_move)) {
    std::cerr << "Error in compact.\n";
    return 6;
  }

  COPCORE_CUDA_CHECK(cudaFree(vect1_ptr_d));
  COPCORE_CUDA_CHECK(cudaFree(vect2_ptr_d));
  COPCORE_CUDA_CHECK(cudaFree(sel_vector_d));

  return 0;
}
