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

The initial container contains as many rays as pixels. In addition, two containers of indices are used. Initially, the first container hold all the indices for initial rays.

The current version of scheduling is: get the ray using the index from the first container of indices, then transport the ray only to the next boundary. If it's still alive, add the index in the second container of indices. After all the rays were transported to the next boundary, swap the pointers of the first and second containers of indices and clear the second container. Finally, add the indices of the reflected rays in the first container of indices. Repeat this at every step. The loop is finished when the number of elements in use from the first container of indices is equal to 0.

## How to use:

```console
$ ./RaytraceBenchmark -gdml_file <path to gdml file>
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
