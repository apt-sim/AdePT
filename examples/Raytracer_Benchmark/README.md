<!--
SPDX-FileCopyrightText: 2020 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# RaytraceBenchmark

## Geometry example using VecGeom

RaytraceBenchmark is a raytracer prototype. The rays are shooted from every "pixel", traverse the geometry model and compute the pixel color based on the model.

The current version uses 2 models:
- transparent with/without reflection (traverse full geometry) 
- specular reflection (stop ray when reaching a specular object)

It currently works for only 2 gdml files: "trackML.gdml" (placed in the source tree of vecgeom) and "box_sphere.gdml" (placed in RaytraceBenchmark directoy).


## Ray transport

Every ray has a parameter called generation. The initial container contains as many rays as pixels and empty containers are created for the reflected rays of higher generations​.

The current version of scheduling is: transport the ray to first boundary only, then generate reflected ray and increase generation, moving to next container. Then, the refracted ray is transported to the next boundary.

The number of generations for the reflected rays is unknown, so 10 containers are created for secondary rays (the rays with generation greater than 10 will be added in the (generation % 10) container). After transporting all the rays from a container, the initial rays are selected and released. This step is repeated for all the containers. In order to be sure that all the secondary rays are transported, the program verifies if there are any secondary rays in the containers. If yes, reiterate over all the containers and transport the rays.


## How to use:

```console
$ ./RaytraceBenchmark -gdml_name <path to gdml file>
```

## Parameters

- on_gpu - run on GPU (default: 1)
- px - Image size in pixels X-axis (default: 1840)
- py - Image size in pixels Y-axis (default: 512)
- view - RT view as in { kRTVparallel = 0, kRTVperspective } (default: kRTVperspective)
- reflection - Use reflection and refraction model (default: 0)
- screenx - Screen position in world coordinates X-axis (default: -5000)
- screeny - Screen position in world coordinates Y-axis (default: 0)
- screenz - Screen position in world coordinates Z-axis (default: 0)
- upx - Up vector X-axis (default: 0)
- upy - Up vector Y-axis (default: 1)
- upz - Up vector Z-axis (default: 0)
- bkgcol - Light color (default: 0xFFFFFF80)
- use_tiles - run on GPU in tiled mode (default: 0) (TODO: not working with the current version)
- block_size - block size for tiled mode (default: 8) (TODO: not working with the current version)



## Light models

The light models are applied at the boundary between 2 volumes, depending on their optical properties.

The light models are:

- transparent with/without reflection and refraction (at the interface between 2 transparent volumes and if their refraction indices are greater than 1 and not equal)
- specular reflection (if the next volume is specular)

The transparent model includes exponential attenuation of rays. Every material has a constant called "transparency per cm" which is the fraction of the incident light passing through after 1 cm of material. Tuning it can make the whole picture more "transparent" or "opaque".


## Volumes

- World: 
	 - material: transparent
	 - object color: 0x0000FF80
	 - refraction index: 1
	 - transparency per cm: 1

- Sphere: 
	- material: transparent
	- object color: 0x0000FF80
	- refraction index: 1.1
	- transparency per cm: 0.982

- Box: 
	- material: kRTspecular
	- object color: 0xFF000080

- trackML objects: 
	- material: transparent
	- object color: 0x0000FF80
	- refraction index: 1.5
	- transparency per cm: 0.7

- Other objects:
	- material: transparent
	- object color: 0x0000FF80
	- refraction index: 1
	- transparency per cm: 1
