<!--
SPDX-FileCopyrightText: 2026 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# AdePT Documentation

[![Nightly CI (EL9)](https://github.com/apt-sim/AdePT/actions/workflows/nightly-ci-el9.yml/badge.svg?branch=master)](https://github.com/apt-sim/AdePT/actions/workflows/nightly-ci-el9.yml)

This site is the home for AdePT installation, user guide, and API reference.

## Introduction

AdePT is a lightweight Geant4 plugin designed to accelerate high-energy
physics simulation by offloading electromagnetic transport to GPUs.
Currently, it targets electrons, positrons, and gammas.

AdePT relies on
[G4HepEm](https://github.com/mnovak42/g4hepem), which provides specialized,
optimized tracking for electrons, positrons, and gammas. AdePT provides the
GPU transport implementation and integrates it with Geant4
applications.

```{toctree}
:maxdepth: 2
:caption: Installation

installation/install-dependencies
installation/build-from-source
```

```{toctree}
:maxdepth: 2
:caption: USER GUIDE

user/gpu_tracking
user/integration-g4
user/runtime-parameters
user/examples
```

```{toctree}
:maxdepth: 2
:caption: API REFERENCE

api/index
```
