#ifndef CUDAHELPERS_H
#define CUDAHELPERS_H

#include <memory>

#define checkCuda( code )                       \
  { assertCuda( code, __FILE__, __LINE__ ); }

inline void assertCuda( cudaError_t code, const char *file, int line, bool abort = true )
{
  if ( code != cudaSuccess )
  {
    printf( "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line );
    if ( abort ) assert( code == cudaSuccess );
  }
}


struct CudaDeleter {
  void operator()(void* mem) {
    checkCuda( cudaFree(mem) );
  }
};

template<typename T>
std::unique_ptr<T, CudaDeleter> make_unique_cuda() {
  T* tmp;
  checkCuda( cudaMalloc(&tmp, sizeof(T)) );
  return {tmp, CudaDeleter()};
}

#endif