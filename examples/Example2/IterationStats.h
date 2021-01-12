#ifndef ITERATIONSTATS_H
#define ITERATIONSTATS_H

//  struct to keep simple statistics on iterations 
//         counters are atomic integers kept in device (global) memory
//         access methods
//  - J. Apostolakis,  7-8  Jan 2021
//
struct IterationStats_dev
{

 public:
   __host__
   __device__
   IterationStats_dev(){ initialise(); }

   __host__   
   __device__
   void initialise() { maxDone.store(0); totalIterations.store(0); }

   // __host__
   __device__
   static IterationStats_dev *MakeInstanceAt(void *addr)
   {
     IterationStats_dev *obj = new (addr) IterationStats_dev();
     // obj->initialise();   //  Add or keep separate ?
     return obj;
   }

   __device__ void updateMax( /*unsigned*/ int iters ){ maxDone.fetch_max( iters); }  // atomicMax( maxdone.data, iters );
   __host__ __device__ void addIters ( /*unsigned*/ int iters ){ totalIterations += iters; }
   
   __host__ __device__ /*unsigned*/ int GetMax()   { return maxDone.load(); }
   __host__ __device__ /*unsigned*/ int GetTotal() { return totalIterations.load(); }

   friend class IterationStats;
   
 private: 
   adept::Atomic_t< /*unsigned*/ int>  maxDone;         // std::atomic<int> on host, adept::AtomicType<int> on device ...
   adept::Atomic_t< /*unsigned*/ int>  totalIterations;
};


__global__ void MakeInstanceAt_glob( char *buffer_dev,
                                     IterationStats_dev** ISdev_PtrHolder ) {
   *ISdev_PtrHolder= IterationStats_dev::MakeInstanceAt(buffer_dev);
}

// Host class for creating and interfacing to 'iteration' statistics on device
// 

class IterationStats {

 public:
   
   __host__ 
   IterationStats()  {
      char *buffer_iterStat = nullptr;
      // cudaMallocManaged(&buffer_iterStat, sizeof(IterationStats));   // In unified memory
      cudaMalloc(&buffer_iterStat, sizeof(IterationStats));   // In device ('global') memory

      // __device__ IterationStats_dev* devicePtrHolder;
      IterationStats_dev** devicePtrHolder;
      cudaMallocManaged(&devicePtrHolder, sizeof(IterationStats_dev*));  // Unified memory
      
      MakeInstanceAt_glob<<<1,1>>>( buffer_iterStat, devicePtrHolder );
      cudaDeviceSynchronize();
      
      // devPtr= ..
      devPtr= *devicePtrHolder;
      // cudaMemcpyFromSymbol( &devPtr, devicePtrHolder, sizeof(IterationStats_dev*) ); // If ptr was in device memory

      assert(devPtr != nullptr);
      cudaFree(devicePtrHolder);
   }
   
   ~IterationStats()  {
      cudaFree(devPtr);
   }

   __host__   /*unsigned*/ int GetMax(){ /*unsigned*/ int maxNow=0;   // Was GetMaxFromDevice()
     cudaMemcpyFromSymbol( &maxNow,
                           &(devPtr->maxDone), sizeof( /*unsigned*/ int) ); return maxNow; }
   
   __host__   void SetMaxIterationsDone(/*unsigned*/ int val) {
     cudaMemcpyToSymbol( &(devPtr->maxDone), &val, sizeof(/*unsigned*/ int) ); // Directly overwrite 'atomic' address.  Ok now!?
     // __device__ unsigned int val_dev;
     // cudaMemcpyToSymbol( val_dev, &val, sizeof(unsigned int) );
     // maxDone.store(val_dev);
   }
   __host__   /*unsigned*/ int GetTotal(){ /*unsigned*/ int totalNow=0;   // Was GetTotalFromDevice()
      cudaMemcpyFromSymbol( &totalNow, &(devPtr->totalIterations), sizeof(/*unsigned*/ int) ); return totalNow; }

   __host__ IterationStats_dev* GetDevicePtr() { return devPtr; }
   
// Enable use in device memory ??   
// __host__ int  GetMaxIterations() { int maxNow=0; cudaMemcpyFromSymbol(&maxNow, maxItersDone_dev, sizeof(int)); return maxNow; }
// __host__ void SetMaxIterations(int maxNow) { cudaMemcpyToSymbol( maxDone_dev, &maxNow, sizeof(int) ); }

   
 private:
   // __device__
   IterationStats_dev* devPtr= nullptr;
};


// Earlier trials
// --------------

// __host__ int  GetMaxIterations( /*__device__*/ const IterationStats & iterStats ) {
//    int mxDev=0; cudaMemcpyFromSymbol(&mxDev, iterStats.maxDone, sizeof(int)); return mxDev; }

// #include "base/AdePT/Atomic.h"
// __device__ adept::Atomic_t<int>   ## No max in Atomic_t !
// __device__ int maxItersDone_dev; // Meant to be visible on both  __device__ __host__
// __device__ int  GetMaxIterationsDone() { return  maxItersDone_dev; }
// __device__ void SetMaxIterationsDone(int iters) { maxItersDone_dev= iters; } 

// #include <AdePT/Atomic.h>
// __device__ adept::Atomic_t<int>  totalIters_dev;
// __device__ adept::Atomic_t<int>  totalIters_dev;   // FAILS -- compiler error : dynamically allocated variable ... 
// __device__ int  GetTotalsIterations() { return  totalIters_dev.load(); }
// __device__ void AddIterations(int iters) { totalIters_dev += iters; } 

// __host__
// int  maxItersDone_host;
// int  GetMaxIterationsDone_host ()          { return maxItersDone_host; }
// void SetMaxIterationsDone_host (int iters) { maxItersDone_host= iters; }

// int  GetMaxIterationsDone_dev() { int mxDev=0; cudaMemcpyFromSymbol( &mxDev, maxItersDone_dev, sizeof(int) ); return mxDev; }
// void SetMaxIterationsDone_dev(int mxDev) { cudaMemcpyToSymbol( maxItersDone_dev, &mxDev, sizeof(int) ); }

// int    GetTotalIterations_dev() { int totItersDev=0; cudaMemcpyFromSymbol( &totItersDev, maxItersDone_dev, sizeof(int) ); return totItersDev; }
// void ReSetTotalIterations_dev() { int zero=0; cudaMemcpyToSymbol( maxItersDone_dev, &zero, sizeof(int) ); }


// __host__ PrepareStatistics

// __host__ void ReportStatistics( IterationStats & iterStats )
// {
  // int  maxChordItersGPU;
  // cudaMemcpy(&maxChordItersGPU, maxItersDone_dev, sizeof(int), cudaMemcpyDeviceToHost);   
// }

#endif
