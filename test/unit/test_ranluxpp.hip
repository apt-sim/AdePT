// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/copcore/Ranluxpp.h>

#include <iostream>

__global__ void kernel(RanluxppDouble *r, double *d, uint64_t *i, double *d2)
{
  *d = r->Rndm();
  *i = r->IntRndm();
  r->Skip(42);
  *d2 = r->Rndm();
}

int main(void)
{
  RanluxppDouble r;
  std::cout << "double: " << r.Rndm() << std::endl;
  std::cout << "int: " << r.IntRndm() << std::endl;

  RanluxppDouble *r_dev;
  cudaMalloc(&r_dev, sizeof(RanluxppDouble));
  double *d_dev_ptr;
  uint64_t *i_dev_ptr;
  double *d2_dev_ptr;
  cudaMalloc(&d_dev_ptr, sizeof(double));
  cudaMalloc(&i_dev_ptr, sizeof(uint64_t));
  cudaMalloc(&d2_dev_ptr, sizeof(double));

  // Transfer the state of the generator to the device.
  cudaMemcpy(r_dev, &r, sizeof(RanluxppDouble), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  kernel<<<1, 1>>>(r_dev, d_dev_ptr, i_dev_ptr, d2_dev_ptr);
  cudaDeviceSynchronize();

  // Generate from the same state on the host.
  double d   = r.Rndm();
  uint64_t i = r.IntRndm();
  r.Skip(42);
  double d2 = r.Rndm();

  // Fetch the numbers from the device for comparison.
  double d_dev;
  uint64_t i_dev;
  double d2_dev;
  cudaMemcpy(&d_dev, d_dev_ptr, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&i_dev, i_dev_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d2_dev, d2_dev_ptr, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int ret = 0;

  std::cout << std::endl;
  std::cout << "double:" << std::endl;
  std::cout << "   host:   " << d << std::endl;
  std::cout << "   device: " << d_dev << std::endl;
  ret += (d != d_dev);

  std::cout << "int:" << std::endl;
  std::cout << "   host:   " << i << std::endl;
  std::cout << "   device: " << i_dev << std::endl;
  ret += (i != i_dev);

  std::cout << "double (after calling Skip(42)):" << std::endl;
  std::cout << "   host:   " << d2 << std::endl;
  std::cout << "   device: " << d2_dev << std::endl;
  ret += (d2 != d2_dev);

  cudaFree(r_dev);
  cudaFree(d_dev_ptr);
  cudaFree(i_dev_ptr);
  cudaFree(d2_dev_ptr);

  return ret;
}
