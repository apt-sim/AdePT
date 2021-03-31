<!--
SPDX-FileCopyrightText: 2020 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# RaytraceBenchmark

Geometry example using VecGeom


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
- use_tiles - run on GPU in tiled mode (default: 0)
- block_size - block size for tiled mode (default: 8)


## Light models

- transparent with/without reflection (including attenuation of rays)
- specular reflection


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
	- transparency per cm: 0.82

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
