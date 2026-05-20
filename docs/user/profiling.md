<!--
SPDX-FileCopyrightText: 2026 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Profiling AdePT

AdePT can be profiled with the NVIDIA tools Nsight Systems (`nsys`) and
Nsight Compute (`ncu`). Note that high level counters are available in the
output when running with `/adept/verbose 4`.

## Nsight Systems

`nsys` is the right tool to inspect whole-application timing, CUDA stream
overlap, kernel launch rates, and which particle type is dominating.

As profiling full production workflows can be difficult,
AdePT provides optional CUDA profiler API hooks so `nsys` can skip GPU geometry
initialization and capture only a transport-loop window. Enable them at build
time with:

```console
cmake -S . -B ./adept-build \
  -DADEPT_COMPUTE_BACKEND=CUDA \
  -DADEPT_ENABLE_NSYS_PROFILING=ON \
  <otherargs>
```

`ADEPT_ENABLE_NSYS_PROFILING` requires the CUDA backend. CMake fails during
configuration if it is requested with a non-CUDA backend.

At runtime, enable the transport capture hook with:

```console
export ADEPT_NSYS_CAPTURE_TRANSPORT=1
```

Optional runtime controls:

| Variable | Default | Meaning |
| --- | :---: | --- |
| `ADEPT_NSYS_CAPTURE_START_AFTER_ITERATIONS` | `0` | Start capture before this transport-loop iteration. |
| `ADEPT_NSYS_CAPTURE_STOP_AFTER_ITERATIONS` | `0` | Stop capture before this transport-loop iteration. `0` means stop during transport teardown. |

The iteration range is half-open: `[start, stop)`. For example,
`START_AFTER_ITERATIONS=50` and `STOP_AFTER_ITERATIONS=150` captures transport
iterations 50 through 149. Invalid or negative values are ignored and treated as
the default `0`.

Run `nsys` with the CUDA profiler API capture range:

```console
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --stats=true --export=sqlite --force-overwrite=true \
  --output adept_transport_profile \
  <application command>
```

Open the report with:

```console
nsys-ui adept_transport_profile.nsys-rep
```

### Plotting AdePT kernel profiles

The exported SQLite file can be summarized with the AdePT profile plotting
script:

```console
python3 scripts/plot_adept_nsys_profile.py \
  --sqlite adept_transport_profile.sqlite \
  --output-prefix adept_transport_profile \
  --title "AdePT"
```

This writes:

- `adept_transport_profile_kernel_profile.png`: total CUDA kernel times,
  transport shares by particle type, the waited kernel category that most often
  reaches the end of an AdePT transport iteration, and the corresponding
  critical-path margin in milliseconds.
- `adept_transport_profile_species_pies.png`: per-species pie charts showing
  which split kernels dominate electron, positron, and gamma transport.
- `adept_transport_profile.txt` plus CSV files with the numeric summaries.

The percentages labelled as "all kernels" are normalized to all CUDA kernels in
the captured range, including injection, population statistics, and bookkeeping.
The percentages labelled as "transport" are normalized only to electron,
positron, and gamma transport kernels.

The limiter plots use the latest-ending waited non-`FinishIteration` CUDA
kernel before `FinishIteration` as the limiting category. `InitTracks` kernels
are excluded from this limiter view because they run on a separate injection
stream and `FinishIteration` does not directly wait for them; they remain visible
in the total CUDA kernel-time view. The count view shows how often each waited
category is last. The critical-margin view credits only the time by which that
latest kernel extends the iteration beyond the runner-up latest kernel, capped
by the latest kernel's own duration.

```{figure} images/nsys_kernel_profile_example.png
:name: fig-nsys-kernel-profile-example
:alt: Example AdePT Nsight Systems kernel profile summary plot.
:align: center
:width: 95%

Example kernel summary from an AdePT split-kernel `nsys` profile.
```

The example above was produced from a CMSSW ttbar simulation. It illustrates why
both the total kernel-time view and the limiter view are useful: gamma transport
accounts for about 25% of all CUDA kernel time and about 27% of transport kernel
time, but gamma kernels almost never determine the end of the transport
iteration in this profile. Electrons and positrons dominate the waited
last-kernel counts, and electrons alone account for most of the critical-path
margin. This means that reducing the gamma workload, for example with Russian
roulette, is not expected to improve the iteration wall time unless the gamma
kernels become the latest waited path. Conversely, `InitTracks` is visible as a
non-negligible total kernel-time bucket, but it is intentionally excluded from
the limiter view because it runs asynchronously on the injection stream.

```{figure} images/nsys_species_pies_example.png
:name: fig-nsys-species-pies-example
:alt: Example AdePT Nsight Systems per-species transport kernel pie charts.
:align: center
:width: 95%

Example per-species transport-kernel summary from the same kind of AdePT
split-kernel `nsys` profile.
```

## Nsight Compute

`ncu` is the tool for detailed kernel analysis: register count, achieved
occupancy, memory coalescing, cache behavior, stall reasons, and source-level
metrics. Use kernel-name filters plus launch skipping/counting to select
representative transport kernels:

```console
ncu --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*Electron(HowFar|Propagation|MSC|Relocation).*' \
  --launch-skip <n> \
  --launch-count <m> \
  --set detailed \
  --force-overwrite \
  --export ncu_electron_geometry \
  <application command>
```

Choose `--launch-skip` from an `nsys` profile so the selected launches are in
the dominant transport region rather than startup or tail iterations. For
source-level attribution, use a build configuration that emits CUDA line
information, such as `RelWithDebInfo`.
