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

- `adept_transport_profile_kernel_profile.png`: total CUDA kernel-time buckets,
  particle transport shares, the kernel category that most often ends an AdePT
  transport iteration, and the same last-kernel category weighted by
  GPU-active iteration time.
- `adept_transport_profile_species_pies.png`: per-species pie charts showing
  which split kernels dominate electron, positron, and gamma transport.
- `adept_transport_profile.txt` plus CSV files with the numeric summaries.

The percentages labelled as "all kernels" are normalized to all CUDA kernels in
the captured range, including injection, population statistics, and bookkeeping.
The percentages labelled as "transport" are normalized only to electron,
positron, and gamma transport kernels.

The limiter plots use the latest-ending non-`FinishIteration` CUDA kernel before
`FinishIteration` as the limiting category. The count-weighted view shows how
often each category is last. The time-weighted view weights the same category by
the GPU-active interval from the first kernel in that paired iteration to the
latest-ending kernel, so expensive iterations matter more than tiny ones. This
identifies which path tends to gate expensive iterations; it is not by itself a
speedup estimate for very short kernels that run after long transport work.
Large gaps between two `FinishIteration` kernels can also be reported in the
text file as diagnostics, but those gaps may include CPU-side work or
profiler/capture latency and should not be interpreted as single GPU transport
iterations.

```{figure} images/nsys_kernel_profile_example.png
:name: fig-nsys-kernel-profile-example
:alt: Example AdePT Nsight Systems kernel profile summary plot.
:align: center
:width: 95%

Example summary from an AdePT split-kernel `nsys` profile.
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
