#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os

import ROOT

INTEGER_HISTOGRAMS = {"generation_population", "primary_ancestor_population"}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot all ROOT truth histograms produced by integrationTest.")
    parser.add_argument("--root-file", required=True, help="Input ROOT file produced by integrationTest --truth_root")
    parser.add_argument("--output-dir", required=True, help="Directory where PNG plots will be written")
    return parser.parse_args()


def read_value_entries(root_file, hist_name, hist):
    # Exact-value histograms are written as "count per distinct floating-point
    # value" plus a sidecar metadata object that maps bins back to the original
    # values. For plotting, reconstruct those value/count pairs first.
    metadata = root_file.Get(hist_name + "__values")
    if not metadata:
        return None
    if metadata.ClassName() != "TObjString" or not hasattr(metadata, "GetString"):
        return None

    text = metadata.GetString().Data().splitlines()
    if len(text) == 1 and text[0] == "__empty__":
        return []

    entries = []
    for bin_index, line in enumerate(text, start=1):
        value = float.fromhex(line)
        count = hist.GetBinContent(bin_index)
        entries.append((value, count))
    return entries


def read_label_entries(root_file, hist_name):
    # Categorical bin labels are stored in sidecar metadata so the test output
    # does not rely on ROOT mutating TAxis label state during the run.
    metadata = root_file.Get(hist_name + "__labels")
    if not metadata:
        return None
    if metadata.ClassName() != "TObjString" or not hasattr(metadata, "GetString"):
        return None

    text = metadata.GetString().Data().splitlines()
    if len(text) == 1 and text[0] == "__empty__":
        return []
    return text


def make_placeholder(name, title, y_title):
    hist = ROOT.TH1D(name, title, 1, 0.0, 1.0)
    hist.SetDirectory(0)
    hist.GetXaxis().SetBinLabel(1, "empty")
    hist.GetYaxis().SetTitle(y_title)
    hist.SetBinContent(1, 0.0)
    return hist


def make_visual_histogram(hist_name, hist, value_entries):
    title = hist.GetTitle()
    if not value_entries:
        return make_placeholder(hist_name + "_plot", title, "Count")

    finite_entries = [(value, count) for value, count in value_entries if math.isfinite(value)]
    if not finite_entries:
        return make_placeholder(hist_name + "_plot", title, "Count")

    min_value = min(value for value, _ in finite_entries)
    max_value = max(value for value, _ in finite_entries)
    unique_values = len(finite_entries)

    if min_value == max_value:
        width = max(abs(min_value) * 1.0e-6, 1.0)
        min_value -= width
        max_value += width

    if hist_name.endswith("num_secondaries"):
        # Integer-like observables are nicer to inspect with unit-width bins.
        min_edge = math.floor(min_value - 0.5)
        max_edge = math.ceil(max_value + 0.5)
        num_bins = max(1, int(max_edge - min_edge))
        visual = ROOT.TH1D(hist_name + "_plot", title, num_bins, min_edge, max_edge)
    else:
        # For dense floating-point observables, compress the exact-value map into
        # a readable visualization while keeping the exact ROOT file unchanged.
        num_bins = min(max(unique_values, 50), 200)
        visual = ROOT.TH1D(hist_name + "_plot", title, num_bins, min_value, max_value)

    visual.SetDirectory(0)
    visual.GetYaxis().SetTitle("Count")
    visual.GetXaxis().SetTitle(hist_name)

    for value, count in finite_entries:
        visual.Fill(value, count)

    return visual


def make_categorical_clone(hist_name, hist, labels):
    clone = hist.Clone(hist_name + "_plot")
    clone.SetDirectory(0)
    clone.GetYaxis().SetTitle("Count" if hist_name != "edep_by_volume" else "Energy Deposit")
    if labels is not None:
        for bin_index in range(1, clone.GetNbinsX() + 1):
            label = "empty"
            if labels and bin_index - 1 < len(labels):
                label = labels[bin_index - 1]
            clone.GetXaxis().SetBinLabel(bin_index, label)
    return clone


def make_numeric_clone(hist_name, hist, y_title):
    clone = hist.Clone(hist_name + "_plot")
    clone.SetDirectory(0)
    clone.GetYaxis().SetTitle(y_title)
    clone.GetXaxis().SetTitle(hist_name)
    return clone


def configure_canvas(hist_name, nbins, is_categorical):
    width = 1600
    if is_categorical:
        width = min(max(1600, 42 * max(nbins, 1)), 7000)

    canvas = ROOT.TCanvas("canvas_" + hist_name, "canvas_" + hist_name, width, 900)
    canvas.SetLeftMargin(0.10)
    canvas.SetRightMargin(0.03)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.30 if is_categorical else 0.14)
    return canvas


def draw_plot(output_dir, hist_name, hist, *, is_categorical, logy):
    canvas = configure_canvas(hist_name, hist.GetNbinsX(), is_categorical)
    if logy:
        canvas.SetLogy(True)
    if hist_name in ("initial_ekin", "final_ekin"):
        # Energy spans several orders of magnitude and is easier to inspect on a
        # logarithmic x-axis.
        canvas.SetLogx(True)

    color = ROOT.kAzure - 3
    if hist_name == "creator_process_counts":
        color = ROOT.kOrange + 7
    elif hist_name == "step_defining_process_counts":
        color = ROOT.kGreen + 2
    elif hist_name == "edep_by_volume":
        color = ROOT.kRed - 4
    elif hist_name.startswith("final_"):
        color = ROOT.kBlue - 7
    elif hist_name.startswith("vertex_"):
        color = ROOT.kMagenta - 3

    hist.SetFillColor(color)
    hist.SetLineColor(ROOT.kBlack)
    hist.SetLineWidth(2)
    hist.SetTitle(hist_name)
    hist.GetXaxis().SetLabelSize(0.028 if is_categorical else 0.032)
    hist.GetYaxis().SetLabelSize(0.032)
    hist.GetYaxis().SetTitleOffset(1.2)

    if is_categorical:
        hist.LabelsOption("v", "X")

    maximum = hist.GetMaximum()
    if maximum <= 0.0:
        maximum = 1.0
    hist.SetMaximum(maximum * (4.0 if logy else 1.25))
    if logy:
        hist.SetMinimum(0.5)

    hist.Draw("hist")
    canvas.SaveAs(os.path.join(output_dir, hist_name + ".png"))


def main():
    args = parse_args()

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(0.035)

    os.makedirs(args.output_dir, exist_ok=True)

    root_file = ROOT.TFile.Open(args.root_file)
    if root_file is None or root_file.IsZombie():
        raise SystemExit(f"Failed to open ROOT file: {args.root_file}")

    hist_names = []
    for key in root_file.GetListOfKeys():
        name = key.GetName()
        if name.endswith("__values"):
            continue
        obj = key.ReadObj()
        if obj.InheritsFrom("TH1"):
            hist_names.append(name)
    hist_names.sort()

    manifest_lines = []
    for hist_name in hist_names:
        hist = root_file.Get(hist_name)
        value_entries = read_value_entries(root_file, hist_name, hist)
        if value_entries is None:
            if hist_name in INTEGER_HISTOGRAMS:
                plot_hist = make_numeric_clone(hist_name, hist, "Count")
                is_categorical = False
            else:
                # Categorical histograms are already directly plottable.
                label_entries = read_label_entries(root_file, hist_name)
                plot_hist = make_categorical_clone(hist_name, hist, label_entries)
                is_categorical = True
        else:
            # Exact-value histograms are converted into a visually readable
            # representation only for plotting.
            plot_hist = make_visual_histogram(hist_name, hist, value_entries)
            is_categorical = False

        draw_plot(args.output_dir, hist_name, plot_hist, is_categorical=is_categorical,
                  logy=(hist_name == "creator_process_counts"))
        manifest_lines.append(hist_name + ".png")

    manifest_path = os.path.join(args.output_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="ascii") as manifest:
        manifest.write("\n".join(manifest_lines) + "\n")

    print(args.output_dir)


if __name__ == "__main__":
    main()
