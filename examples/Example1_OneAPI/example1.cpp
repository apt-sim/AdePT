// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>

int main() {
   // Creating buffer of 4 ints to be used inside the kernel code
   cl::sycl::buffer<cl::sycl::cl_int, 1> Buffer(4);

   // Creating SYCL queue
   cl::sycl::queue Queue;

   cl::sycl::range<1> NumOfWorkItems{Buffer.get_count()};


   Queue.submit([&](cl::sycl::handler &cgh) {

    auto Accessor = Buffer.get_access<cl::sycl::access::mode::write>(cgh);
        // Executing kernel
	cgh.parallel_for<class FillBuffer>(
	NumOfWorkItems, [=](cl::sycl::id<1> WIid) {
	// Fill buffer with indexes
	Accessor[WIid] = (cl::sycl::cl_int)WIid.get(0);
	});
    });

   const auto HostAccessor = Buffer.get_access<cl::sycl::access::mode::read>();

   bool MismatchFound = false;
   for (size_t I = 0; I < Buffer.get_count(); ++I) {
     if (HostAccessor[I] != I) {
         std::cout << "The result is incorrect for element: " << I
                  << " , expected: " << I << " , got: " << HostAccessor[I] << std::endl;
         MismatchFound = true;
         }
    }

   if (!MismatchFound) {
       std::cout << "The results are correct!" << std::endl;
   }
   std::cout << "Device: " << Queue.get_device().get_info<cl::sycl::info::device::name>() << std::endl;
   return MismatchFound;
}
