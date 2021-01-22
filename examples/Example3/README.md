<!--
SPDX-FileCopyrightText: 2020 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

Example3 README
===============

This code is work in progress. The idea with this is to focus on
learning how to fit HEP simulations on the GPU with capabilities more or
less similar to Geant4, so that a real project can be created later with
better insights on how things need to work to achieve reasonable
performance. The other point is to learn what is the best overall
structure of the code, good ways to try to achieve maintainability and
portability at least across CPUs and GPUs, and how different parts of
the code will need to communicate with each other to allow enough
flexibility later on to correct problems in design at ealier stages of
development. For these reasons, classes and templates are kept to the
absolute minimum, headers cannot at this point contain any concrete
implementations of the modules they serve, everything that can be local
to a file (like functions) are declared static and not available in any
headers, simple algorithms are used where they can be easily replaced
later by more sophisticated ones, etc. Each module is currently a cpp
file and a matching header, but in a full project these could be, for
instance, a set of classes and headers that become a full library. Many
compromises have been made in order to have a minimally functional
prototype as soon as possible in order to experiment with it. For
example, the way particles are handled on GPU and how alive particles
are counted, or how particles are stored in AoS format at the moment.
Later on, the idea is to change storage layout of particles into one
buffer per set of related components and have different sets of buffers
for each type of particle. Similarly, although all particles are stepped
together now, stepping with more particle types when they are introduced
will have to work in a different way. The stepping function will need to
be split into several parts, the common ones being run for all
particles, like integration in magnetic field for all charged particles,
and checking distance to next boundary, whereas physics processes will
be in different kernels for each particle type. Process selection will
likely have to be coordinated across different kernels as well, and use
some storage per track, something that doesn't happen now. The random
number generator states are currently managed independently of the rest
of the track states, and will likely have to be managed together in the
future as well. The important property to keep for now is that changes
to how random numbers are managed should not affect any code calling the
APIs (like uniform(a,b)) for generating a random number. This is what
affords flexibility to change the random number generation code
independently from the rest later, without needing to change physics
models, etc. The last point is that currently the CUDA code requires a
lot of #ifdefs that will hopefully be eliminated in later revisions by
introducing new functions and separating better host and device code
wherever possible. After this revision is finished, it will be time to
add more realistic physics using real particles like photons and
electrons at low energy and testing, e.g. energy deposition vs depth in
a semi-infinite slab of Si or other similar simple problem. Then, we
could add support for more complex geometries, introduce proper
materials, acceleration structures for navigation, ray casting functions
to support at least the most common GDML shapes, and finally run
benchmarks to compare performance with Geant4 for this simple problem.
At this point we should have a good idea of how to proceed with full
simulation on GPUs and this code can either be discarded or evolved into
a full project, whichever makes more sense.
