#!/usr/bin/env python3
"""
plot.py: visualization utility for particle trajectories.

Input text file format:

label1
N1
x_11 y_11 z_11 px_11 py_11 pz_11 s_11
...
label2
N2
...

Each row represents (x, y, z, px, py, pz, s).
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

###############################################################################
# Internal utilities
###############################################################################


def set_axes_equal(ax):
    """Forces an isotropic reference frame for a 3D plot."""
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()

    ranges = [abs(b - a) for a, b in (x_lim, y_lim, z_lim)]
    centers = [(a + b) / 2.0 for a, b in (x_lim, y_lim, z_lim)]
    radius = max(ranges) * 0.5

    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


class Trajectory:
    """Container for a trajectory"""

    def __init__(self, data: np.ndarray, label: str):
        self.data = data  # shape (N, 7)
        self.label = label

    # Accès commodité
    @property
    def x(self):
        return self.data[:, 0]

    @property
    def y(self):
        return self.data[:, 1]

    @property
    def z(self):
        return self.data[:, 2]

    @property
    def px(self):
        return self.data[:, 3]

    @property
    def py(self):
        return self.data[:, 4]

    @property
    def pz(self):
        return self.data[:, 5]

    @property
    def s(self):
        return self.data[:, 6]

    def components(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Dict {comp: (s, values)} for x, y, z, px, py, pz."""
        return {
            "x": (self.s, self.x),
            "y": (self.s, self.y),
            "z": (self.s, self.z),
            "px": (self.s, self.px),
            "py": (self.s, self.py),
            "pz": (self.s, self.pz),
        }


def resample_component(s_old: np.ndarray, val_old: np.ndarray, s_new: np.ndarray):
    """Linear interpolation on a curvilinear grid."""
    f = interp1d(s_old, val_old, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(s_new)


def resample_trajectory(traj: Trajectory, s_new: np.ndarray) -> Trajectory:
    """Returns a resampled trajectory on s_new."""
    comps = [resample_component(traj.s, traj.data[:, i], s_new) for i in range(6)]
    stacked = np.column_stack(comps + [s_new])
    return Trajectory(stacked, traj.label)


def load_trajectories(path: Path, pmax: int | None = None) -> Dict[str, Trajectory]:
    """Read a file containing trajectories, with optional resampling."""
    trajs: Dict[str, Trajectory] = {}

    with path.open("r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        label = lines[i]
        n_points = int(lines[i + 1])
        raw_block = lines[i + 2 : i + 2 + n_points]
        i += 2 + n_points

        data = np.array([[float(v) for v in row.split()] for row in raw_block])

        if pmax and n_points > pmax:
            step = math.ceil(n_points / pmax)
            data = data[::step]

        trajs[label] = Trajectory(data, label)

    if not trajs:
        raise ValueError("The file does not contain any trajectory.")

    return trajs


###############################################################################
# Sub-command: traj
###############################################################################


def plot_trajectories(trajs: Dict[str, Trajectory], what: str):
    """Plots 3D positions (what=='pos') or 3D momentums (what=='mom')."""
    coord = ("x", "y", "z") if what == "pos" else ("px", "py", "pz")
    labels = ("x", "y", "z") if what == "pos" else ("px", "py", "pz")

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1,1,1])

    for traj in trajs.values():
        xs, ys, zs = (getattr(traj, c) for c in coord)
        ax.plot(xs, ys, zs, label=traj.label)

        # Trajectory origin
        ax.scatter(xs[0], ys[0], zs[0], c="r", s=10)

    set_axes_equal(ax)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title("Trajectoires " + ("position" if what == "pos" else "momentum"))
    ax.legend()
    return fig


###############################################################################
# Sub-command: errs
###############################################################################


def collect_ref_test_pairs(
    trajs: Dict[str, Trajectory], base_label: str
) -> List[Tuple[Trajectory, Trajectory]]:
    """
    Returns [(ref, test), ...] where labels are formatted as
    <base_label>*_ref and <base_label>*_test.
    """
    pairs = []
    for label in trajs:
        if label.startswith(base_label) and label.endswith("_ref"):
            root = label[: -4]  # sans '_ref'
            test_label = root + "_test"
            if test_label in trajs:
                pairs.append((trajs[label], trajs[test_label]))
    if not pairs:
        raise ValueError(f"No ref/test pairs found with prefix {base_label}.")
    return pairs


def mean_error_fig(
    traj_pairs: List[Tuple[Trajectory, Trajectory]], use_relative: bool, n_pts: int = 100
):
    """2x3 figure of means and standard deviations at 1,3,5 sigma."""
    comps = ("x", "y", "z", "px", "py", "pz")
    err_arrays = {c: [] for c in comps}

    # grille s commune à toutes les paires (intersection des intervalles valides)
    s_start = max(pair[0].s[0] for pair in traj_pairs)
    s_end = min(pair[0].s[-1] for pair in traj_pairs)
    s_grid = np.linspace(s_start, s_end, n_pts)

    for ref, tst in traj_pairs:
        ref_rs = resample_trajectory(ref, s_grid)
        tst_rs = resample_trajectory(tst, s_grid)

        for c in comps:
            ref_vals = getattr(ref_rs, c)
            tst_vals = getattr(tst_rs, c)
            if use_relative:
                denom = np.where(ref_vals == 0.0, 1.0, np.abs(ref_vals))
                err = np.abs(ref_vals - tst_vals) / denom
            else:
                err = np.abs(ref_vals - tst_vals)
            err_arrays[c].append(err)

    # passage en numpy
    for k in comps:
        err_arrays[k] = np.vstack(err_arrays[k])  # shape (Npairs, Npoints)

    # tracé
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()

    for idx, c in enumerate(comps):
        ax = axes[idx]
        data = err_arrays[c]
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        # mini = data.min(axis=0)
        # maxi = data.max(axis=0)

        ax.plot(s_grid, mean, label="average")
        # ax.plot(s_grid, mini, label="min")
        # ax.plot(s_grid, maxi, label="max")
        for sigma in (1, 3, 5):
            ax.fill_between(
                s_grid,
                mean - sigma * std,
                mean + sigma * std,
                alpha=0.2 / sigma,
                label=rf"{sigma}$\sigma$",
            )

        ax.set_title(c)
        ax.set_xlabel("s")
        ax.set_ylabel("error")
        if idx == 0:
            ax.legend()

    fig.suptitle(("Relative" if use_relative else "Absolute") + " error")
    return fig


###############################################################################
# Sub-command: comp
###############################################################################


def compare_errors_fig(
    trajs: Dict[str, Trajectory],
    ref_label: str,
    use_relative: bool,
    sel_labels: List[str] | None,
    n_pts: int = 500,
):
    """2x3 figures of individual plots of error against reference."""
    if ref_label not in trajs:
        raise ValueError(f"Reference label {ref_label} not found in file.")

    comps = ("x", "y", "z", "px", "py", "pz")
    ref_traj = trajs[ref_label]

    # Sélection des trajs à comparer
    if sel_labels:
        test_labels = [l for l in sel_labels if l != ref_label]
    else:
        test_labels = [l for l in trajs if l != ref_label]

    if not test_labels:
        raise ValueError("No trajectories to compare.")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()

    for tst_label in test_labels:
        tst_traj = trajs[tst_label]

        # grille s commune
        s_start = max(ref_traj.s[0], tst_traj.s[0])
        s_end = min(ref_traj.s[-1], tst_traj.s[-1])
        s_grid = np.linspace(s_start, s_end, n_pts)

        ref_rs = resample_trajectory(ref_traj, s_grid)
        tst_rs = resample_trajectory(tst_traj, s_grid)

        for idx, c in enumerate(comps):
            ref_vals = getattr(ref_rs, c)
            tst_vals = getattr(tst_rs, c)
            if use_relative:
                denom = np.where(ref_vals == 0.0, 1.0, np.abs(ref_vals))
                err = np.abs(ref_vals - tst_vals) / denom
            else:
                err = np.abs(ref_vals - tst_vals)

            axes[idx].plot(s_grid, err, label=tst_label)

    for idx, c in enumerate(comps):
        ax = axes[idx]
        ax.set_title(c)
        ax.set_xlabel("s")
        ax.set_ylabel("error")
        if idx == 0:
            ax.legend()

    fig.suptitle(
        f"{'Relative' if use_relative else 'Absolute'} error vs reference '{ref_label}'"
    )
    return fig


###############################################################################
# Sub-commands: multierrs
###############################################################################


def mean_multi_error_fig(
    traj_pairs_dict: Dict[str, List[Tuple[Trajectory, Trajectory]]], use_relative: bool, n_pts: int = 500
):
    """Plots errors accross multiple sets of ref/test trajectories in the same plots."""
    comps = ("x", "y", "z", "px", "py", "pz")
    err_arrays = {c: {name: [] for name in traj_pairs_dict.keys()} for c in comps}

    # grille s commune à toutes les paires (intersection des intervalles valides)
    s_start = max(max(pair[0].s[0] for pair in traj_pairs) for traj_pairs in traj_pairs_dict.values())
    s_end = min(min(pair[0].s[-1] for pair in traj_pairs)  for traj_pairs in traj_pairs_dict.values())
    s_grid = np.linspace(s_start, s_end, n_pts)

    for pairs_name, traj_pairs in traj_pairs_dict.items():
        for ref, tst in traj_pairs:
            ref_rs = resample_trajectory(ref, s_grid)
            tst_rs = resample_trajectory(tst, s_grid)

            for c in comps:
                ref_vals = getattr(ref_rs, c)
                tst_vals = getattr(tst_rs, c)
                if use_relative:
                    denom = np.where(ref_vals == 0.0, 1.0, np.abs(ref_vals))
                    err = np.abs(ref_vals - tst_vals) / denom
                else:
                    err = np.abs(ref_vals - tst_vals)
                err_arrays[c][pairs_name].append(err)

    # passage en numpy
    for k in comps:
        for pairs_name in err_arrays[k].keys():
            err_arrays[k][pairs_name] = np.vstack(err_arrays[k][pairs_name])  # shape (Npairs, Npoints)

    # tracé
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()

    for idx, c in enumerate(comps):
        for pairs_name in err_arrays[c].keys():
            ax = axes[idx]
            data = err_arrays[c][pairs_name]
            mean = data.mean(axis=0)

            ax.plot(s_grid, mean, label=f"\"{pairs_name}\" average")

            ax.set_title(c)
            ax.set_xlabel("s")
            ax.set_ylabel("error")
            if idx == 0:
                ax.legend()
            # ax.set_ylim(-0.01, 3)

    fig.suptitle(("Relative" if use_relative else "Absolute") + " error")
    return fig


###############################################################################
# CLI
###############################################################################


def build_cli():
    parser = argparse.ArgumentParser(
        description="Trajectory plotting utilities"
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # traj
    p_traj = sub.add_parser("traj", help="3D plot of trajectories")
    p_traj.add_argument("file", type=Path, help="Input text file")
    grp = p_traj.add_mutually_exclusive_group()
    grp.add_argument("-p", "--pos", action="store_true", help="Plot positions (default)")
    grp.add_argument("-m", "--mom", action="store_true", help="Plot momentums")
    p_traj.add_argument("--pmax", type=int, help="Max number of points to load per trajectories")

    # errs
    p_errs = sub.add_parser("errs", help="Error statistics ref/test")
    p_errs.add_argument("file", type=Path)
    p_errs.add_argument("base_label", help="Common prefix ref/test trajectories")
    grp2 = p_errs.add_mutually_exclusive_group(required=True)
    grp2.add_argument("-a", "--abs", action="store_true", help="Absolute error")
    grp2.add_argument("-r", "--rel", action="store_true", help="Relative error")
    p_errs.add_argument("--pmax", type=int, help="Max number of points to load per trajectories")

    # comp
    p_comp = sub.add_parser("comp", help="Plots all errors to a single reference trajectory")
    p_comp.add_argument("file", type=Path)
    p_comp.add_argument("ref_label", help="Label of the reference")
    grp3 = p_comp.add_mutually_exclusive_group(required=True)
    grp3.add_argument("-a", "--abs", action="store_true")
    grp3.add_argument("-r", "--rel", action="store_true")
    p_comp.add_argument(
        "-l",
        "--labels",
        nargs="+",
        help="Labels to compare and plot errors of, otherwise all in the file",
    )
    p_comp.add_argument("--pmax", type=int, help="Max number of points to load per trajectories")

    # multierrs
    p_multierrs = sub.add_parser("multierrs", help="Error statistics ref/test between multiple files")
    p_multierrs.add_argument("files", nargs="+", type=Path)
    p_multierrs.add_argument(
        "-b",
        "--base-labels",
        nargs="+",
        help="Common prefix to ref/test, one for each file",
        required=True,
    )
    p_multierrs.add_argument(
        "-n",
        "--names",
        nargs="+",
        help="Labels of each file in the plots",
        required=True)
    grp4 = p_multierrs.add_mutually_exclusive_group(required=True)
    grp4.add_argument("-a", "--abs", action="store_true", help="Absolute error")
    grp4.add_argument("-r", "--rel", action="store_true", help="Relative error")
    p_multierrs.add_argument("--pmax", type=int, help="Max number of points to load per trajectories")

    return parser


def main():
    parser = build_cli()
    args = parser.parse_args()

    if args.cmd == "traj":
        trajs = load_trajectories(args.file, args.pmax) if "file" in args else {}
        what = "mom" if args.mom else "pos"
        fig = plot_trajectories(trajs, what)
        plt.show()

    elif args.cmd == "errs":
        trajs = load_trajectories(args.file, args.pmax) if "file" in args else {}
        pairs = collect_ref_test_pairs(trajs, args.base_label)
        fig = mean_error_fig(pairs, use_relative=args.rel)
        plt.show()

    elif args.cmd == "comp":
        trajs = load_trajectories(args.file, args.pmax) if "file" in args else {}
        fig = compare_errors_fig(
            trajs,
            ref_label=args.ref_label,
            use_relative=args.rel,
            sel_labels=args.labels,
        )
        plt.show()
    
    elif args.cmd == "multierrs":
        pairs_dict = {}
        for file, base_label, pairs_name in zip(args.files, args.base_labels, args.names):
            trajs = load_trajectories(file, args.pmax)
            pairs = collect_ref_test_pairs(trajs, base_label)
            pairs_dict[pairs_name] = pairs
        fig = mean_multi_error_fig(pairs_dict, use_relative=args.rel)
        plt.show()


if __name__ == "__main__":
    main()
