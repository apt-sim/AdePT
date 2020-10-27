#include "render.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv)
{
	if (argc != 4) {
		std::cout << "Usage: " << argv[0] << " height width output.ppm" << std::endl;
		return 0;
	}

	int height = std::atoi(argv[1]);
	int width = std::atoi(argv[2]);

	render(argv[3], height, width);

	return 0;
}
