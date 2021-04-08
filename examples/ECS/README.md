<!--
SPDX-FileCopyrightText: 2021 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# ECS Example

This example compares an ECS/SoA data layout with AoS. The trick about an ECS is that things that are fields
of a struct/class in classical C++ become components of Archetypes. Those components are array-like, and algorithms
work on components instead of single structs.

An Archetype (= particle type in our case) is a struct of multiple components, and when particles of different Archetypes
are processed, one just reads the components that are required for a specific algorithm. In this example, this is achieved
using a struct of references, which selects the components necessary for a specific algorithm. For simplicity, all components
are selected, but this isn't a requirement. One could equally well create structs of references that only select a subset of
components, such that one can write algorithms that work on e.g. the yellow and red fields in the figure below.

![image](https://user-images.githubusercontent.com/249404/106000510-2d456d00-60af-11eb-9e26-06e3113e2abe.png)

`ecs.h` defines a dummy track struct whose size can be chosen at compile time. For a GPU,
I envision something like 256 tracks in one block (a high multiple of the warp size).
When setting the size to 1, one recovers a classic AoS-style track.

To run some tests, a bunch of TrackBlocks are collected in a slab of memory. A proper implementation will of course be
a bit smarter than just a multi-dimensional array:
- Blocks might be assigned to processing queues
- Blocks might be full / partly occupied / empty

Some ideas are listed in [AdePT#72](https://github.com/apt-sim/AdePT/issues/72).

## Workflows
The example runs a few different workflows that use progressively more fields of the track.
- Enumerating particles. This write to a single field.
- Initialise position and momentum. Use those to compute a mock energy.
- Use RNG to transport particles by random distances.
- As above, but kill particles with low probability to provoke holes
- As above, but merge blocks that have low occupancy.

Findings are discussed in https://indico.cern.ch/event/1020971/.

## Note on launch kernels
Note that launchers or more complicated than necessary to switch between AoS and SoA:
```c++
template<typename Slab>
__global__ void run_compute_energy(Slab* slab, unsigned int loadFactor = 1)
{
  for (unsigned int taskIdx = 0; taskIdx < slab->tasks_per_slab; ++taskIdx) {
    for (unsigned int globalTrackIdx_inTask = blockIdx.x * blockDim.x + threadIdx.x;
        globalTrackIdx_inTask < Slab::tracks_per_block * Slab::blocks_per_task;
        globalTrackIdx_inTask += gridDim.x * blockDim.x) {

      const auto tIdx = globalTrackIdx_inTask % Slab::tracks_per_block;
      const auto bIdx = globalTrackIdx_inTask / Slab::tracks_per_block;

      init_pos_mom(slab->tracks[taskIdx][bIdx][tIdx]);
      compute_energy(slab->tracks[taskIdx][bIdx][tIdx]);
    }
  }
}
```
They can eventually collapse to something like
```c++
template<typename Slab>
__global__ void run_compute_energy(Slab* slab, unsigned int loadFactor = 1)
{
  for (unsigned int taskIdx = 0; taskIdx < slab->tasks_per_slab; ++taskIdx) {
    for (unsigned int bIdx = blockIdx.x; bIdx < Slab::blocks_per_task; bIdx += gridDim.x) {
	  for (unsigned int tIdx = threadIdx.x; tIdx < Slab::threads_per_block; tIdx += blockDim.x) {
       	init_pos_mom(slab->tracks[taskIdx][bIdx], tIdx);
       	compute_energy(slab->tracks[taskIdx][bIdx], tIdx);
	  }
    }
  }
}
```
when the switching between SoA/AoS is removed from the workflow.


## Example output, Tesla V100
```
ECS construct slabs           	      47.303 ms.
AoS construct slabs           	      53.865 ms.
cuda malloc+memcpy            	      78.396 ms.
ECS enumerate_particles       	       0.973 ms.	 GPU                          	       0.072 ms.	 GPU (lambda, run 1x)         	       0.022 ms.	 GPU (lambda, run 100x)       	       0.394 ms.	
AoS enumerate_particles       	      13.640 ms.	 GPU                          	       0.342 ms.	 GPU (lambda, run 1x)         	       0.349 ms.	 GPU (lambda, run 100x)       	      33.816 ms.	
ECS seed rng                  	     136.442 ms.
AoS seed rng                  	     125.092 ms.
ECS compute_energy            	      11.662 ms.	 GPU (run 1x, invoke as kernel)	       0.121 ms.	 GPU (run 1x, invoke as lambda)	       0.076 ms.	 GPU (run 100x, invoke as lambda)	       7.346 ms.	
AoS compute_energy            	      18.010 ms.	 GPU (run 1x, invoke as kernel)	       1.776 ms.	 GPU (run 1x, invoke as lambda)	       1.771 ms.	 GPU (run 100x, invoke as lambda)	     220.164 ms.	
Particles from ECS, CPU:
	id=       0 x=(  0.0000   0.1000   0.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=0 0 ...
	id=       1 x=(  1.0000   1.1000   1.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=0 0 ...
	id=     258 x=(258.0000 258.1000 258.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=0 0 ...
	id=  131331 x=(259.0000 259.1000 259.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=0 0 ...
Particles from AoS, CPU:
	id=       0 x=(  0.0000   0.1000   0.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=0 0 ...
	id=       1 x=(  1.0000   1.1000   1.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=0 0 ...
	id=     258 x=(258.0000 258.1000 258.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=0 0 ...
	id=  131331 x=(259.0000 259.1000 259.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=0 0 ...
Memcpy back                   	      76.458 ms.
Particles from ECS, GPU:
	id=       0 x=(  0.0000   0.1000   0.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=1 0 ...
	id=       1 x=(  1.0000   1.1000   1.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=9F1C67142C84C502 24D94E3C4B490E8 ...
	id=     258 x=(258.0000 258.1000 258.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=147DE65B54EC4CB 391D163471025E13 ...
	id=  131331 x=(259.0000 259.1000 259.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=DD50544B1B156CD2 6529D80FF7DE5872 ...
Particles from AoS, GPU:
	id=       0 x=(  0.0000   0.1000   0.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=1 0 ...
	id=       1 x=(  1.0000   1.1000   1.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=9F1C67142C84C502 24D94E3C4B490E8 ...
	id=     258 x=(258.0000 258.1000 258.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=147DE65B54EC4CB 391D163471025E13 ...
	id=  131331 x=(259.0000 259.1000 259.2000) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=DD50544B1B156CD2 6529D80FF7DE5872 ...
Memcpy GPU to GPU             	       0.758 ms.

ECS advance_by_random_distance	      16.849 ms.	 GPU (run 1x)                 	       0.545 ms.	 GPU (run 100x)               	      53.072 ms.	 GPU (run 100x, create holes) 	      41.342 ms.
AoS advance_by_random_distance	      19.506 ms.	 GPU (run 1x)                 	       0.945 ms.	 GPU (run 100x)               	      96.745 ms.	 GPU (run 100x, create holes) 	      57.439 ms.
 GPU (run separate launches 100x, create holes, compact)	      39.671 ms.
 GPU (run separate launches 100x, create holes, compact)	      60.576 ms.
Memcpy back                   	      76.055 ms.
Particles from ECS, GPU:
	id=       0 x=( 49.9417  99.9833 150.0250) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=5934FACED2DFCC77 D9CB7DDEA0E485B6 ...
	id=  655607 x=(300.2117 353.5232 406.8351) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=B2CF2FD99CD6D276 DE5A127A8FBDA82C ...
	id=  131423 x=(405.5286 460.1566 514.7849) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=CDADD7FA52AFB345 ED0DA2F695E67611 ...
	id= -131331 x=(277.6599 296.4198 315.1797) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=2BBD9AA6BE3E60EF 7745E972CDFDC157 ...
Particles from AoS, GPU:
	id=       0 x=( 49.9417  99.9833 150.0250) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=5934FACED2DFCC77 D9CB7DDEA0E485B6 ...
	id=-1179707 x=(101.1339 143.3678 185.6017) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=95E647C8BF733DE 660F700A9504F614 ...
	id= 1048915 x=(389.2711 439.6429 490.0141) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=CE5AEA53C56236F9 AC20B541714D1C3D ...
	id= -131331 x=(277.6599 296.4198 315.1797) v=(  1.0000   2.0000   3.0000) E=  3.7417 rand=2BBD9AA6BE3E60EF 7745E972CDFDC157 ...
```
