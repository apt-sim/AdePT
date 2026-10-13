#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0
"""Sketches for the geometry validation of the region handoff and of the GPU boundary crossings
(docs/user/geometry-validation.md).

Writes docs/user/images/geometry_validation_{handoff,crossings,overlaps}.png.
Tolerance bands are drawn hugely exaggerated.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "user", "images")
GPU = "#f4b183"   # GPU-region volume
CPU = "#dbe8f5"   # CPU-region volume
BAND = "#9e9e9e"  # tolerance band
OK, BAD, LEGIT = "#2e7d32", "#c62828", "#6a1b9a"


def arrow(ax, p, q, color="k", lw=1.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, linestyle=ls))


def boundary(ax, x=5.0, band=0.25, ymin=0.6, ymax=5.4):
    ax.add_patch(Rectangle((x - band, ymin), 2 * band, ymax - ymin, color=BAND, alpha=0.35, lw=0))
    ax.plot([x, x], [ymin, ymax], color="k", lw=1.2)


def frame(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")


def two_volumes(ax, left="GPU-region volume\n(being left)", right="CPU-region volume\n(VecGeom next state)"):
    ax.add_patch(Rectangle((0.3, 0.6), 4.7, 4.8, color=GPU, lw=0))
    ax.add_patch(Rectangle((5.0, 0.6), 4.7, 4.8, color=CPU, lw=0))
    ax.text(1.0, 4.9, left, fontsize=8, va="top")
    ax.text(5.6, 4.9, right, fontsize=8, va="top")
    boundary(ax)


def handoff_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    # (a) confirmed
    ax = axes[0, 0]
    frame(ax, "(a) confirmed")
    two_volumes(ax)
    arrow(ax, (2.0, 1.6), (5.0, 3.0))
    ax.plot(5.0, 3.0, "o", color=OK, ms=6)
    arrow(ax, (5.0, 3.0), (7.2, 4.0), color=OK)
    arrow(ax, (5.0, 3.0), (6.2, 3.0), color="k", lw=1, style="-|>")
    ax.text(6.25, 2.75, "n", fontsize=9)
    ax.text(5.3, 1.0, "Geant4 relocates from the VecGeom path\nwith the final direction: path confirmed",
            fontsize=8, color=OK)
    ax.text(4.3, 5.55, "Geant4 band (exaggerated)", fontsize=7, color="#555")
    # (b) direction reversed at the boundary
    ax = axes[0, 1]
    frame(ax, "(b) direction reversed at the boundary: legitimate")
    two_volumes(ax)
    arrow(ax, (2.0, 1.6), (5.0, 3.0))
    ax.plot(5.0, 3.0, "o", color=LEGIT, ms=6)
    arrow(ax, (5.0, 3.0), (3.4, 4.1), color=LEGIT)
    arrow(ax, (5.0, 3.0), (6.2, 3.0), color="k", lw=1)
    ax.text(6.25, 2.75, "n", fontsize=9)
    ax.text(5.3, 0.9, "deflection (MSC, field) after the\ncrossing: n·d < 0 at the exit point,\nGeant4 puts the track "
            "back in the\nGPU volume (native Geant4: a zero\nstep to the same place)", fontsize=8, color=LEGIT)
    # (c) coincident surface
    ax = axes[1, 0]
    frame(ax, "(c) coincident surface: legitimate")
    two_volumes(ax)
    ax.add_patch(Rectangle((5.0, 2.2), 0.9, 1.6, color="#90caf9", lw=0))
    ax.text(5.1, 3.85, "X", fontsize=9)
    arrow(ax, (2.0, 1.6), (5.0, 3.0))
    ax.plot(5.0, 3.0, "o", color=LEGIT, ms=6)
    arrow(ax, (5.0, 3.0), (7.0, 3.9), color=LEGIT)
    ax.text(5.6, 1.0, "a further volume X starts on the\nsame surface; the direction enters it:\n"
            "Geant4 locates X (a zero step on the GPU)", fontsize=8, color=LEGIT)
    # (d) errors
    ax = axes[1, 1]
    frame(ax, "(d) errors: band violation, unexplained relocation")
    two_volumes(ax)
    arrow(ax, (2.0, 1.6), (4.1, 2.6))
    ax.plot(4.1, 2.6, "o", color=BAD, ms=6)
    ax.annotate("", xy=(4.1, 2.25), xytext=(5.0, 2.25), arrowprops=dict(arrowstyle="<->", color=BAD, lw=1))
    ax.text(3.9, 1.9, "beyond the band", fontsize=7, color=BAD, ha="right")
    arrow(ax, (4.1, 2.6), (6.5, 3.6), color="#555", ls="--")
    ax.text(5.3, 0.95, "end point strictly inside the volume being\nleft (or outside the one entered): band error.\n"
            "Located elsewhere without (b) or (c):\nunexplained, unless an overlap is proven",
            fontsize=8, color=BAD)
    fig.suptitle("Geometry validation, /adept/validateHandoff: tracks leaving the GPU regions, relocated as Geant4 resumes tracking",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometry_validation_handoff.png"), dpi=130)


def crossings_figure():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    frame(ax, "")
    ax.set_xlim(0, 14)
    ax.add_patch(Rectangle((0.3, 0.6), 6.7, 4.8, color=GPU, lw=0, alpha=0.7))
    ax.add_patch(Rectangle((7.0, 0.6), 6.7, 4.8, color="#ffe0b2", lw=0))
    ax.text(0.6, 5.1, "volume left", fontsize=9)
    ax.text(10.6, 5.1, "volume entered", fontsize=9)
    ax.add_patch(Rectangle((6.6, 0.6), 0.8, 4.8, color=BAND, alpha=0.4, lw=0))
    ax.plot([7, 7], [0.6, 5.4], "k", lw=1.2)
    ax.text(7.0, 5.55, "±kTolerance band (exaggerated)", fontsize=8, ha="center", color="#555")
    pts = [((6.85, 4.3), OK, "in band: depth 0 or (0, 1] kTolerance\n(histogram in kTolerance units)"),
           ((5.6, 3.4), BAD, "kind 1: more than kTolerance inside\nthe volume being left"),
           ((8.6, 2.5), BAD, "kind 2: more than kTolerance outside\nthe volume being entered")]
    for (x, y), c, t in pts:
        ax.plot(x, y, "o", color=c, ms=7)
        ax.text(x + (0.35 if x > 7 else -0.35), y, t, fontsize=8, color=c, va="center", ha="left" if x > 7 else "right")
    ax.plot(3.0, 1.3, "o", color=BAD, ms=7)
    ax.text(3.35, 1.3, "kind 3: farther than kTolerance\nfrom every crossed surface", fontsize=8, color=BAD,
            va="center")
    # bounce-back
    arrow(ax, (7.0, 1.7), (8.2, 1.7), color="k", lw=1)
    ax.text(8.25, 1.55, "n", fontsize=9)
    arrow(ax, (7.0, 1.7), (5.9, 2.3), color=LEGIT)
    ax.text(5.8, 2.45, "bounce-back:\nfinal d · n < 0", fontsize=8, color=LEGIT, ha="right")
    fig.suptitle("Geometry validation, /adept/validateCrossings: every boundary crossing relocated on the GPU (solid navigation)",
                 fontsize=11)
    ax.text(0.4, 0.15, "Off-band landings are recorded with the point and direction in the frame of the offending "
            "volume (replay on its solid) and in the global frame.", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometry_validation_crossings.png"), dpi=130)


def overlaps_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    frame(ax, "(a) overlap between branches")
    ax.add_patch(Rectangle((0.6, 0.8), 5.0, 4.2, color=GPU, alpha=0.8, lw=0))
    ax.add_patch(Rectangle((4.4, 1.6), 5.0, 3.0, color="#90caf9", alpha=0.7, lw=0))
    ax.add_patch(Rectangle((4.4, 1.6), 1.2, 3.0, fill=False, hatch="///", lw=0, color="#444"))
    ax.text(0.9, 4.6, "A (left)", fontsize=9)
    ax.text(7.6, 4.2, "B (entered)", fontsize=9)
    arrow(ax, (1.6, 2.0), (4.4, 3.0))
    ax.plot(4.4, 3.0, "o", color=BAD, ms=6)
    ax.text(4.3, 3.3, "p", fontsize=9, ha="right")
    ax.plot(5.0, 3.21, "s", color=LEGIT, ms=6)
    ax.annotate("witness q = p + δ·d", xy=(5.0, 3.21), xytext=(6.2, 1.0), fontsize=8, color=LEGIT,
                arrowprops=dict(arrowstyle="-", color=LEGIT, lw=0.8))
    ax.text(0.6, 0.2, "p on B's surface, strictly inside A; the witness q is strictly\ninside A and B, which belong to "
            "different branches: proven overlap", fontsize=8)
    ax = axes[1]
    frame(ax, "(b) extrusion")
    ax.add_patch(Rectangle((0.6, 0.8), 6.0, 4.2, color=GPU, alpha=0.5, lw=0))
    ax.add_patch(Rectangle((5.2, 2.0), 3.4, 1.6, color="#ef9a9a", alpha=0.9, lw=0))
    ax.text(0.9, 4.6, "mother M", fontsize=9)
    ax.text(5.4, 3.7, "daughter D", fontsize=9)
    ax.plot(7.6, 2.8, "s", color=LEGIT, ms=6)
    ax.text(7.7, 2.3, "point", fontsize=8, color=LEGIT)
    ax.text(0.6, 0.2, "a point strictly inside D and strictly outside its\nancestor M: proven extrusion of D out of M",
            fontsize=8)
    fig.suptitle("Geometry validation, overlap proof from the solids: errors explained by the geometry are listed per pair",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometry_validation_overlaps.png"), dpi=130)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    handoff_figure()
    crossings_figure()
    overlaps_figure()
