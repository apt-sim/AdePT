# AdePT
Accelerated demonstrator of electromagnetic Particle Transport

## Build Requirements
The following packages are a required to build and run:

- CMake >= 3.17
- C/C++ Compiler with C++14 support
- CUDA Toolkit (tested 10.1, min version TBD)

To build, simply run:

```console
$ cmake -S. -B./adept-build <otherargs>
...
$ cmake --build ./adept-build
```
