// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef STEPPING_H
#define STEPPING_H

class particle;

void step_all_particles();
void transport(particle &);

#endif
