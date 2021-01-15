#ifndef ITERATIONSTATS_H
#define ITERATIONSTATS_H

//  struct to keep simple statistics on iterations 
//         counters are atomic integers kept in device (global) memory
//         access methods
//  - J. Apostolakis,  7-8  Jan 2021
//
struct IterationStats_impl
{
 // Implementation class - can live in host or device

 public:
   __host__
   __device__
   IterationStats_impl(){ initialise(); }

   __host__   
   __device__
   void initialise() { maxDone.store(0); totalIterations.store(0); }

   // __host__
   __device__
   static IterationStats_impl *MakeInstanceAt(void *addr)
   {
     IterationStats_impl *obj = new (addr) IterationStats_impl();
     // obj->initialise();   //  Add or keep separate ?
     return obj;
   }

   __device__ void updateMax( /*unsigned*/ int iters ){ maxDone.fetch_max( iters); }
   // __host__   void updateMax( /*unsigned*/ int iters ){ }  // Implementation needed! 
   __host__ __device__ void addIters ( /*unsigned*/ int iters ){ totalIterations += iters; }
   
   __host__ __device__ /*unsigned*/ int GetMax()   { return maxDone.load(); }
   __host__ __device__ /*unsigned*/ int GetTotal() { return totalIterations.load(); }

   friend class IterationStats;
   
 private: 
   adept::Atomic_t< /*unsigned*/ int>  maxDone;         // std::atomic<int> on host, adept::AtomicType<int> on device ...
   adept::Atomic_t< /*unsigned*/ int>  totalIterations;
};


__global__ void MakeInstanceAt_glob( char *buffer_dev,
                                     IterationStats_impl** ISdev_PtrHolder ) {
   *ISdev_PtrHolder= IterationStats_impl::MakeInstanceAt(buffer_dev);
}

__global__ void Initialise_glob( IterationStats_impl* iterStats_dev ) 
{
   iterStats_dev->initialise();
}

// Host class for creating and interfacing to 'iteration' statistics on device
// 

class IterationStats {

 public:
   
   __host__ 
   IterationStats()  {
      // Simplification suggested by Jonas H. 2021.01.14
      cudaMalloc(&this->devPtr, sizeof(IterationStats_impl));   // In device ('global') memory
      // cudaMallocManaged(&this->devPtr, sizeof(IterationStats_impl));   // In device ('global') memory
      Initialise_glob<<<1,1>>>( this->devPtr );
   }
   
   ~IterationStats()  {
      cudaFree(devPtr);
   }

   __host__   /*unsigned*/ int GetMax(){ /*unsigned*/ int maxNow=0;   // Was GetMaxFromDevice()
     cudaMemcpy(&maxNow, &(devPtr->maxDone), sizeof( /*unsigned*/ int), cudaMemcpyDeviceToHost);   
     return maxNow; }
      // cudaMemcpyFromSymbol( &maxNow, (devPtr->maxDone), sizeof( /*unsigned*/ int) ); return maxNow; }
   
   __host__   void SetMaxIterationsDone(/*unsigned*/ int val) {
     // Directly overwrite 'atomic' address -- is correct-ness implementation dependent ?
      cudaMemcpy( &(devPtr->maxDone), &val, sizeof(/*unsigned*/ int ), cudaMemcpyHostToDevice);
     // cudaMemcpyToSymbol( &(devPtr->maxDone), &val, sizeof(/*unsigned*/ int) );
   }
   __host__   /*unsigned*/ int GetTotal(){ /*unsigned*/ int totalNow=0;   // Was GetTotalFromDevice()
      cudaMemcpy( &totalNow, &(devPtr->totalIterations), sizeof(/*unsigned*/ int), cudaMemcpyDeviceToHost); 
      // cudaMemcpyFromSymbol( &totalNow, devPtr->totalIterations, sizeof(/*unsigned*/ int) );
   return totalNow; }

   __host__ IterationStats_impl* GetDevicePtr() { return devPtr; }
   
 private:
   // __device__
   IterationStats_impl* devPtr= nullptr;  // 
};

#endif
