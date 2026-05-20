#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ITERATION_ORDER = (
    "electron",
    "positron",
    "gamma",
    "gamma WDT",
    "track injection",
    "track enqueue",
    "population stats",
    "bookkeeping",
    "VecGeom init",
    "other",
)

PARTICLE_ORDER = ("electron", "positron", "gamma")


def ns_to_ms(value):
    return value / 1.0e6


def pct(value, total):
    return 100.0 * value / total if total else 0.0


def is_finish_iteration(name):
    return "FinishIteration" in name


def classify_transport_kernel(name):
    if "TransportElectrons" in name:
        return "electron", "Transport"
    if "TransportPositrons" in name:
        return "positron", "Transport"
    if "TransportGammasWoodcock" in name:
        return "gamma", "GammaWoodcock"
    if "TransportGammas" in name:
        return "gamma", "Transport"

    if "Gamma" in name:
        if "GammaWoodcock" in name:
            return "gamma", "GammaWoodcock"
        if "GammaPropagation" in name:
            return "gamma", "Propagation"
        if "GammaHowFar" in name:
            return "gamma", "HowFar"
        if "GammaRelocation" in name:
            return "gamma", "Relocation"
        if any(token in name for token in (
            "GammaSetupInteractions",
            "GammaConversion",
            "GammaCompton",
            "GammaPhotoelectric",
        )):
            return "gamma", "Interactions"
        return "gamma", "Interactions"

    if "Electron" in name and "<(bool)1" in name:
        particle = "electron"
    elif ("Electron" in name and "<(bool)0" in name) or "Positron" in name:
        particle = "positron"
    else:
        return None, None

    if "ElectronPropagation" in name:
        return particle, "Propagation"
    if "ElectronHowFar" in name:
        return particle, "HowFar"
    if "ElectronMSC" in name:
        return particle, "MSC"
    if "ElectronRelocation" in name:
        return particle, "Relocation"
    if any(token in name for token in (
        "ElectronSetupInteractions",
        "ElectronBremsstrahlung",
        "ElectronIonization",
        "PositronStoppedAnnihilation",
        "PositronAnnihilation",
    )):
        return particle, "Interactions"
    return particle, "Interactions"


def classify_kernel_bucket(name):
    particle, detail = classify_transport_kernel(name)
    if particle:
        if detail == "GammaWoodcock":
            return "gamma WDT", "transport: gamma WDT", particle, detail
        return particle, f"transport: {particle}", particle, detail
    if "InitTracks" in name:
        return "track injection", "track injection", None, None
    if "EnqueueTracks" in name:
        return "track enqueue", "track enqueue", None, None
    if "CountCurrentPopulation" in name or "ZeroEventCounters" in name:
        return "population stats", "population stats", None, None
    if any(token in name for token in (
        "FinishIteration",
        "FreeSlots",
        "ClearAllQueues",
        "InitSlotManagers",
        "InitParticleQueues",
        "InitQueue",
    )):
        return "bookkeeping", "bookkeeping", None, None
    if "ConstructOnGpu" in name or "ConstructManyOnGpu" in name:
        return "VecGeom init", "VecGeom init", None, None
    return "other", "other", None, None


def read_kernel_rows(sqlite_path):
    query = """
        SELECT k.start, k.end, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """
    uri = f"file:{Path(sqlite_path).resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        yield from conn.execute(query)


def finalize_iteration(rows, finish_start, finish_end, limiter_counts, limiter_kernel_ns, limiter_critical_ns,
                       limiter_durations, finish_gap_ns, idle_gap_threshold_ns):
    if not rows:
        return
    candidates = [
        row for row in rows
        if not is_finish_iteration(row[4])
        and row[3] != "track injection"
        and row[1] <= finish_start
    ]
    if not candidates:
        return
    limiter = max(candidates, key=lambda row: row[1])
    bucket = limiter[3]
    duration = limiter[2]
    limiter_counts[bucket] += 1
    limiter_kernel_ns[bucket] += duration
    limiter_durations.append(duration)

    # Approximate the critical-path margin of the last kernel. If a tiny
    # bookkeeping kernel happens to run after a long transport phase, it should
    # only receive credit for the time by which it actually extends the end of
    # the iteration, not for all preceding transport work.
    sorted_by_end = sorted(candidates, key=lambda row: row[1])
    runner_up_end = sorted_by_end[-2][1] if len(sorted_by_end) > 1 else limiter[0]
    critical_ns = min(duration, max(0, limiter[1] - runner_up_end))
    limiter_critical_ns[bucket] += critical_ns

    first_start = min(row[0] for row in candidates)
    active_ns = limiter[1] - first_start
    finish_gap_ns.append(active_ns)


def summarize(sqlite_path, idle_gap_threshold_ms):
    bucket_ns = defaultdict(int)
    species_detail_ns = {particle: defaultdict(int) for particle in PARTICLE_ORDER}
    top_kernel_ns = defaultdict(int)
    top_kernel_calls = defaultdict(int)
    limiter_counts = defaultdict(int)
    limiter_kernel_ns = defaultdict(int)
    limiter_critical_ns = defaultdict(int)
    limiter_durations = []
    finish_gap_ns = []

    current_iteration = []
    previous_finish_end = -1
    idle_gap_threshold_ns = int(idle_gap_threshold_ms * 1.0e6)

    for start, end, _stream, name in read_kernel_rows(sqlite_path):
        duration = end - start
        iteration_bucket, time_bucket, particle, detail = classify_kernel_bucket(name)
        bucket_ns[time_bucket] += duration
        top_kernel_ns[name] += duration
        top_kernel_calls[name] += 1
        if particle:
            species_detail_ns[particle][detail] += duration

        if start <= previous_finish_end:
            continue
        current_iteration.append((start, end, duration, iteration_bucket, name))
        if is_finish_iteration(name):
            finalize_iteration(current_iteration, start, end, limiter_counts, limiter_kernel_ns, limiter_critical_ns,
                               limiter_durations, finish_gap_ns, idle_gap_threshold_ns)
            previous_finish_end = end
            current_iteration = []

    return bucket_ns, species_detail_ns, limiter_counts, limiter_kernel_ns, limiter_critical_ns, limiter_durations, finish_gap_ns, top_kernel_ns, top_kernel_calls


def collapsed_particle_limiter(limiter_counts, limiter_kernel_ns):
    counts = defaultdict(int)
    kernel_ns = defaultdict(int)
    for bucket, count in limiter_counts.items():
        particle = "gamma" if bucket == "gamma WDT" else bucket
        if particle in PARTICLE_ORDER:
            counts[particle] += count
            kernel_ns[particle] += limiter_kernel_ns[bucket]
    return counts, kernel_ns


def write_csvs(output_prefix, bucket_ns, species_detail_ns, limiter_counts, limiter_kernel_ns, limiter_critical_ns):
    total_kernel_ns = sum(bucket_ns.values())
    transport_ns = sum(sum(values.values()) for values in species_detail_ns.values())
    total_iterations = sum(limiter_counts.values())
    total_limiter_kernel_ns = sum(limiter_kernel_ns.values())
    total_limiter_critical_ns = sum(limiter_critical_ns.values())

    bucket_csv = output_prefix.with_name(output_prefix.name + "_kernel_buckets.csv")
    with bucket_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "kernel_ms", "all_kernel_fraction"])
        for bucket, value in sorted(bucket_ns.items(), key=lambda item: item[1], reverse=True):
            writer.writerow([bucket, ns_to_ms(value), pct(value, total_kernel_ns) / 100.0])

    species_csv = output_prefix.with_name(output_prefix.name + "_species_breakdown.csv")
    with species_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["particle", "kernel", "kernel_ms", "species_fraction", "transport_fraction", "all_kernel_fraction"])
        for particle in PARTICLE_ORDER:
            species_total = sum(species_detail_ns[particle].values())
            for detail, value in sorted(species_detail_ns[particle].items(), key=lambda item: item[1], reverse=True):
                writer.writerow([
                    particle,
                    detail,
                    ns_to_ms(value),
                    pct(value, species_total) / 100.0,
                    pct(value, transport_ns) / 100.0,
                    pct(value, total_kernel_ns) / 100.0,
                ])

    limiter_csv = output_prefix.with_name(output_prefix.name + "_limiter.csv")
    with limiter_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "limiter",
            "iterations",
            "iteration_fraction",
            "limiter_kernel_ms",
            "limiter_kernel_fraction",
            "critical_margin_ms",
            "critical_margin_fraction",
        ])
        for bucket in ITERATION_ORDER:
            if limiter_counts[bucket] == 0 and limiter_kernel_ns[bucket] == 0:
                continue
            writer.writerow([
                bucket,
                limiter_counts[bucket],
                pct(limiter_counts[bucket], total_iterations) / 100.0,
                ns_to_ms(limiter_kernel_ns[bucket]),
                pct(limiter_kernel_ns[bucket], total_limiter_kernel_ns) / 100.0,
                ns_to_ms(limiter_critical_ns[bucket]),
                pct(limiter_critical_ns[bucket], total_limiter_critical_ns) / 100.0,
            ])

    particle_counts, particle_kernel_ns = collapsed_particle_limiter(limiter_counts, limiter_kernel_ns)
    particle_csv = output_prefix.with_name(output_prefix.name + "_particle_limiter.csv")
    particle_iterations = sum(particle_counts.values())
    particle_kernel_total = sum(particle_kernel_ns.values())
    with particle_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["particle", "iterations", "particle_limiter_fraction", "limiter_kernel_ms", "particle_limiter_kernel_fraction"])
        for particle in PARTICLE_ORDER:
            writer.writerow([
                particle,
                particle_counts[particle],
                pct(particle_counts[particle], particle_iterations) / 100.0,
                ns_to_ms(particle_kernel_ns[particle]),
                pct(particle_kernel_ns[particle], particle_kernel_total) / 100.0,
            ])


def write_summary(output_prefix, title, bucket_ns, species_detail_ns, limiter_counts, limiter_kernel_ns,
                  limiter_critical_ns, limiter_durations, finish_gap_ns, top_kernel_ns, top_kernel_calls, idle_gap_threshold_ms):
    total_kernel_ns = sum(bucket_ns.values())
    transport_by_particle = {particle: sum(values.values()) for particle, values in species_detail_ns.items()}
    transport_ns = sum(transport_by_particle.values())
    total_iterations = sum(limiter_counts.values())
    total_limiter_kernel_ns = sum(limiter_kernel_ns.values())
    total_limiter_critical_ns = sum(limiter_critical_ns.values())
    particle_counts, particle_kernel_ns = collapsed_particle_limiter(limiter_counts, limiter_kernel_ns)
    particle_iterations = sum(particle_counts.values())
    particle_kernel_total = sum(particle_kernel_ns.values())
    idle_gap_threshold_ns = idle_gap_threshold_ms * 1.0e6
    large_finish_gaps = [value for value in finish_gap_ns if value > idle_gap_threshold_ns]

    txt_path = output_prefix.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write(f"Total summed kernel time: {ns_to_ms(total_kernel_ns):.3f} ms\n")
        handle.write(f"Summed transport kernel time: {ns_to_ms(transport_ns):.3f} ms\n")
        handle.write(f"Transport iterations paired: {total_iterations}\n")
        if limiter_durations:
            handle.write(f"Max limiter-kernel duration: {ns_to_ms(max(limiter_durations)):.3f} ms\n")
        if finish_gap_ns:
            handle.write(
                f"Max finish-paired active span: {ns_to_ms(max(finish_gap_ns)):.3f} ms "
                f"({len(large_finish_gaps)} spans > {idle_gap_threshold_ms:g} ms; diagnostic only)\n"
            )
        handle.write("\n")

        handle.write("Particle transport share:\n")
        for particle in PARTICLE_ORDER:
            value = transport_by_particle[particle]
            handle.write(
                f"  {particle:8s} {ns_to_ms(value):12.3f} ms"
                f"  {pct(value, total_kernel_ns):6.2f}% all kernels"
                f"  {pct(value, transport_ns):6.2f}% transport\n"
            )

        handle.write("\nKernel-time buckets:\n")
        for bucket, value in sorted(bucket_ns.items(), key=lambda item: item[1], reverse=True):
            handle.write(f"  {bucket:24s} {ns_to_ms(value):12.3f} ms  {pct(value, total_kernel_ns):6.2f}%\n")

        handle.write("\nLimiter categories, latest-ending non-FinishIteration kernel before FinishIteration:\n")
        for bucket in ITERATION_ORDER:
            count = limiter_counts[bucket]
            value = limiter_kernel_ns[bucket]
            active = limiter_critical_ns[bucket]
            if count == 0 and value == 0 and active == 0:
                continue
            handle.write(
                f"  {bucket:16s} {count:8d} iters  {pct(count, total_iterations):6.2f}%"
                f"  {ns_to_ms(active):12.3f} ms  {pct(active, total_limiter_critical_ns):6.2f}% critical margin"
                f"  ({ns_to_ms(value):.3f} ms direct limiter-kernel time)\n"
            )

        handle.write("\nParticle limiter only, gamma WDT counted as gamma:\n")
        for particle in PARTICLE_ORDER:
            count = particle_counts[particle]
            value = particle_kernel_ns[particle]
            handle.write(
                f"  {particle:8s} {count:8d} iters  {pct(count, particle_iterations):6.2f}%"
                f"  {ns_to_ms(value):12.3f} ms  {pct(value, particle_kernel_total):6.2f}% particle limiter-kernel time\n"
            )

        handle.write("\nTop kernels:\n")
        for name, value in sorted(top_kernel_ns.items(), key=lambda item: item[1], reverse=True)[:25]:
            handle.write(f"  {ns_to_ms(value):12.3f} ms  {top_kernel_calls[name]:8d} calls  {name}\n")

    write_csvs(output_prefix, bucket_ns, species_detail_ns, limiter_counts, limiter_kernel_ns, limiter_critical_ns)
    return txt_path


def draw_pie(ax, title, values):
    values = {key: value for key, value in values.items() if value > 0}
    if not values:
        ax.text(0.5, 0.5, "no matching kernels", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return
    items = sorted(values.items(), key=lambda item: item[1], reverse=True)
    labels = [key for key, _ in items]
    sizes = [value for _, value in items]
    total = sum(sizes)
    wedges, _texts, _autotexts = ax.pie(sizes, autopct="%1.1f%%", startangle=90, pctdistance=0.72,
                                        textprops={"fontsize": 8})
    legend_labels = [f"{label}: {ns_to_ms(value):.1f} ms ({pct(value, total):.1f}%)"
                     for label, value in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.36),
              fontsize=7, frameon=False)
    ax.set_title(title)


def draw_summary_plot(output_png, title, bucket_ns, species_detail_ns, limiter_counts, limiter_kernel_ns,
                      limiter_critical_ns):
    transport_by_particle = {particle: sum(values.values()) for particle, values in species_detail_ns.items()}
    transport_ns = sum(transport_by_particle.values())
    total_iterations = sum(limiter_counts.values())
    total_limiter_critical_ns = sum(limiter_critical_ns.values())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{title} - GPU Kernel Profile", fontsize=16)

    ax = axes[0, 0]
    buckets = dict(sorted(bucket_ns.items(), key=lambda item: item[1], reverse=True))
    labels = list(buckets)
    values = [ns_to_ms(buckets[label]) for label in labels]
    ax.barh(labels[::-1], values[::-1])
    ax.set_xlabel("summed kernel time [ms]")
    ax.set_title("All CUDA kernels")

    ax = axes[0, 1]
    labels = list(PARTICLE_ORDER)
    values = [transport_by_particle[label] for label in labels]
    ax.pie(values, labels=[f"{label}\n{pct(transport_by_particle[label], transport_ns):.1f}%" for label in labels],
           startangle=90, textprops={"fontsize": 9})
    ax.set_title("Transport kernels by particle")

    ax = axes[1, 0]
    limiter_labels = [bucket for bucket in ITERATION_ORDER if limiter_counts[bucket] or limiter_kernel_ns[bucket]]
    iter_counts = [limiter_counts[bucket] for bucket in limiter_labels]
    ax.bar(limiter_labels, iter_counts)
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylim(0, max(iter_counts) * 1.25 if iter_counts else 1)
    ax.set_ylabel("iterations")
    ax.set_title("Latest waited category before FinishIteration")

    ax = axes[1, 1]
    critical_ms = [ns_to_ms(limiter_critical_ns[bucket]) for bucket in limiter_labels]
    ax.bar(limiter_labels, critical_ms)
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylim(0, max(critical_ms) * 1.25 if critical_ms else 1)
    ax.set_ylabel("critical-path margin [ms]")
    ax.set_title("Critical-path margin of latest waited category")

    fig.tight_layout()
    fig.savefig(output_png, dpi=180)


def draw_species_plot(output_png, title, bucket_ns, species_detail_ns):
    total_kernel_ns = sum(bucket_ns.values())
    transport_ns = sum(sum(values.values()) for values in species_detail_ns.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"{title} - Transport Kernel Time Per Species", fontsize=16)
    for ax, particle in zip(axes, PARTICLE_ORDER):
        particle_total = sum(species_detail_ns[particle].values())
        draw_pie(
            ax,
            f"{particle.capitalize()}\n{pct(particle_total, total_kernel_ns):.1f}% all kernels, {pct(particle_total, transport_ns):.1f}% transport",
            species_detail_ns[particle],
        )
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(output_png, dpi=180)


def main():
    parser = argparse.ArgumentParser(description="Plot a generic AdePT nsys split-kernel profile.")
    parser.add_argument("--sqlite", required=True, help="Exported nsys SQLite file")
    parser.add_argument("--output-prefix", required=True, help="Output prefix for PNG/TXT/CSV files")
    parser.add_argument("--title", default="AdePT", help="Title prefix for plots and summaries")
    parser.add_argument("--idle-gap-threshold-ms", type=float, default=1000.0,
                        help="Threshold for reporting large finish-paired spans as diagnostics")
    args = parser.parse_args()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize(args.sqlite, args.idle_gap_threshold_ms)
    txt_path = write_summary(output_prefix, args.title, *summary, args.idle_gap_threshold_ms)
    summary_png = output_prefix.with_name(output_prefix.name + "_kernel_profile.png")
    species_png = output_prefix.with_name(output_prefix.name + "_species_pies.png")
    draw_summary_plot(summary_png, args.title, summary[0], summary[1], summary[2], summary[3], summary[4])
    draw_species_plot(species_png, args.title, summary[0], summary[1])

    print(f"Wrote {summary_png}")
    print(f"Wrote {species_png}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
