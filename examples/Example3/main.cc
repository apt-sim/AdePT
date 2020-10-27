#include "common.h"
#include "loop.h"

#include <iostream>

int main(int argc, char **argv)
{
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " nparticles" << std::endl;
		return 0;
	}

	int nparticles = std::atoi(argv[1]);

#ifndef __CUDACC__
	simulate(nparticles);
#else
	simulate_all(nparticles);
#endif

	return 0;
}
