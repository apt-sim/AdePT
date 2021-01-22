// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "loop.h"

#include "init.h"
#include "particle.h"
#include "transport.h"
#include "user.h"

void simulate(int n)
{
	user_init();

	init(n);

	particle p;
	while (get_next_particle(p)) {
		event_init();
		transport(p);
		event_exit();
	}

	user_exit();
}

void simulate_all(int n)
{
	user_init();

	init(n);

	while (particles_alive() > 0)
		step_all_particles();

	user_exit();
}
