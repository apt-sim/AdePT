// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "ppm.h"

#include <fstream>

void write_ppm(const char *name, int *buf, int w, int h)
{
	std::ofstream image(name);

	image << "P3\n" << w << " " << h << "\n255\n";

	for (int j = h - 1; j >= 0; --j) {
		for (int i = 0; i < w; ++i) {
			int idx = j * w + i;
			int b = 0xff & buf[idx];
			int g = 0xff & (buf[idx] >> 8);
			int r = 0xff & (buf[idx] >> 16);
			image << r << " " << g << " " << b << "\n";
		}
	}
}
