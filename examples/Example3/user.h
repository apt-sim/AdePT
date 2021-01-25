// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef USER_H
#define USER_H

#include "common.h"

class particle;

void user_init();
void user_exit();
void event_init();
void event_exit();

__host__ __device__ void track_init(particle &p);
__host__ __device__ void track_exit(particle &p);
__host__ __device__ void step_init(particle &p);
__host__ __device__ void step_exit(particle &p);

#endif
