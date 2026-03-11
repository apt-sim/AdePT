// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/copcore/Ranluxpp.h>

#include <iostream>

#include <AdePT/core/Portability.hh>

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
  ADEPT_DEVICE_API_CALL(Malloc)(&r_dev, sizeof(RanluxppDouble));
  double *d_dev_ptr;
  uint64_t *i_dev_ptr;
  double *d2_dev_ptr;
  ADEPT_DEVICE_API_CALL(Malloc)(&d_dev_ptr, sizeof(double));
  ADEPT_DEVICE_API_CALL(Malloc)(&i_dev_ptr, sizeof(uint64_t));
  ADEPT_DEVICE_API_CALL(Malloc)(&d2_dev_ptr, sizeof(double));

  // Transfer the state of the generator to the device.
  ADEPT_DEVICE_API_CALL(Memcpy)(r_dev, &r, sizeof(RanluxppDouble), ADEPT_DEVICE_API_SYMBOL(MemcpyHostToDevice));
  ADEPT_DEVICE_API_CALL(DeviceSynchronize)();

  kernel<<<1, 1>>>(r_dev, d_dev_ptr, i_dev_ptr, d2_dev_ptr);
  ADEPT_DEVICE_API_CALL(DeviceSynchronize)();

  // Generate from the same state on the host.
  double d   = r.Rndm();
  uint64_t i = r.IntRndm();
  r.Skip(42);
  double d2 = r.Rndm();

  // Fetch the numbers from the device for comparison.
  double d_dev;
  uint64_t i_dev;
  double d2_dev;
  ADEPT_DEVICE_API_CALL(Memcpy)(&d_dev, d_dev_ptr, sizeof(double), ADEPT_DEVICE_API_SYMBOL(MemcpyDeviceToHost));
  ADEPT_DEVICE_API_CALL(Memcpy)(&i_dev, i_dev_ptr, sizeof(uint64_t), ADEPT_DEVICE_API_SYMBOL(MemcpyDeviceToHost));
  ADEPT_DEVICE_API_CALL(Memcpy)(&d2_dev, d2_dev_ptr, sizeof(double), ADEPT_DEVICE_API_SYMBOL(MemcpyDeviceToHost));
  ADEPT_DEVICE_API_CALL(DeviceSynchronize)();

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

  ADEPT_DEVICE_API_CALL(Free)(r_dev);
  ADEPT_DEVICE_API_CALL(Free)(d_dev_ptr);
  ADEPT_DEVICE_API_CALL(Free)(i_dev_ptr);
  ADEPT_DEVICE_API_CALL(Free)(d2_dev_ptr);

  return ret;
}
