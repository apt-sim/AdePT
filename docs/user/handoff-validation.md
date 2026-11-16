<!--
SPDX-FileCopyrightText: 2026 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Handoff validation

When only some regions are transported on the GPU (see
{ref}`runtime-parameters-specify-the-regions-where-the-gpu-is-used`), a track
leaving them is returned to Geant4 with a touchable built from the VecGeom state
reached on the GPU, and Geant4 relocates it from that path with its final
direction. If Geant4 finds a different volume, the track can bounce between CPU
and GPU or continue in the wrong volume.

```console
/adept/validateHandoff true
```

checks every track that leaves the GPU regions on a boundary: a separate Geant4
navigator repeats the relocation, and the outcome is classified. The check only
observes; the simulation is unchanged. When it is off, it costs one pointer
test per returned track.

```{figure} images/handoff_validation.png
:name: fig-validation-handoff
:alt: Outcomes of the handoff of tracks leaving the GPU regions.
:align: center
:width: 95%

Outcomes of the handoff check.
```

| Outcome | Condition |
| --- | --- |
| confirmed | Geant4 relocates into the path reached on the GPU, and no surface lies within the push (Geant4's `ComputeStep` from the end point is not shorter than the push) |
| coincident surface | Geant4 goes deeper, into a volume whose surface holds the end point and that the direction enters (native Geant4 takes a zero step into it), and no surface lies within the push |
| inconclusive: crossed by the handoff push | Geant4 accepts the GPU path, or a coincident volume, at the end point, but a surface lies within the push of 1e-7 mm along the direction applied to returned tracks; what it skips (a thin volume, the world boundary) is not checked |
| inconclusive: overlap | an overlap or extrusion is proven at the end point, or within 100 `kTolerance` along the direction |
| inconclusive: replicas | a parameterised or replicated volume is in the paths; their solids are not checked |
| unexpected: band error | the end point is strictly inside a volume being left, strictly outside a volume being entered, or on none of the crossed surfaces (Geant4's `Inside`) |
| unexpected: direction reversed | the final direction points back through the crossed surface. Electrons and positrons in this situation are kept on the GPU, so the GPU turn-back check was skipped (`ADEPT_USE_SURF`) or its VecGeom normal disagrees with Geant4's |
| unexpected: other path | Geant4 relocates elsewhere without one of the reasons above |

At the end of each run, each worker thread prints its counts and its verdict:

- `HandoffValidation: PASSED`: no unexpected handoff, and at least one handoff
  was conclusive;
- `HandoffValidation: FAILED`: at least one unexpected handoff; the first ten
  are printed with the volumes, the end point and the direction at full
  precision;
- `HandoffValidation: NO CONCLUSIVE HANDOFF`: no track left the GPU regions in
  this thread, or all its handoffs were inconclusive.

A job passes if no thread prints `FAILED` and at least one prints `PASSED`. The
`example1-handoff` test (label `validation`) runs the CMS 2018 geometry with
only `EcalRegion` on the GPU, 2 threads, and checks this.
