#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

"""Plot the geometry validation output of AdePT (/adept/geometryValidationOutput <prefix>).

Reads <prefix>_crossings.json (/adept/validateCrossings) and <prefix>_handoff.json (/adept/validateHandoff), writes
the histograms to <output-dir>/<name>_geometry_validation.root and one PNG per histogram, in the style of
plot_truth_histograms.py.
"""

import argparse
import array
import json
import os

import ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the AdePT geometry validation output.")
    parser.add_argument("--prefix", required=True, help="Output prefix given to /adept/geometryValidationOutput")
    parser.add_argument("--output-dir", required=True, help="Directory where the ROOT file and PNG plots are written")
    parser.add_argument("--name", default=None, help="Name of the setup in titles and file names (default: prefix)")
    return parser.parse_args()


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def configure_canvas(name, *, logx=False, logy=True, categorical=False):
    canvas = ROOT.TCanvas("canvas_" + name, "canvas_" + name, 1600, 900)
    canvas.SetLeftMargin(0.10)
    canvas.SetRightMargin(0.03)
    canvas.SetTopMargin(0.08)
    canvas.SetBottomMargin(0.30 if categorical else 0.14)
    canvas.SetLogx(logx)
    canvas.SetLogy(logy)
    return canvas


def draw(output_dir, hist, color, *, logx=False, logy=True, categorical=False, lines=()):
    canvas = configure_canvas(hist.GetName(), logx=logx, logy=logy, categorical=categorical)
    hist.SetFillColor(color)
    hist.SetLineColor(ROOT.kBlack)
    hist.SetLineWidth(2)
    hist.GetXaxis().SetLabelSize(0.035 if categorical else 0.032)
    hist.GetYaxis().SetLabelSize(0.032)
    hist.GetYaxis().SetTitleOffset(1.2)
    hist.GetXaxis().SetTitleOffset(3.4 if categorical else 1.2)
    hist.GetXaxis().CenterTitle(True)
    if categorical:
        hist.LabelsOption("v", "X")
    maximum = max(hist.GetMaximum(), 1.0)
    hist.SetMaximum(maximum * (4.0 if logy else 1.25))
    if logy:
        hist.SetMinimum(0.5)
    hist.Draw("hist text0" if categorical else "hist")
    keep = []
    for x, color_line in lines:
        line = ROOT.TLine(x, 0.5 if logy else 0.0, x, hist.GetMaximum())
        line.SetLineStyle(2)
        line.SetLineWidth(2)
        line.SetLineColor(color_line)
        line.Draw()
        keep.append(line)
    canvas.SaveAs(os.path.join(output_dir, hist.GetName() + ".png"))
    return keep


def categorical(name, title, labels, counts):
    hist = ROOT.TH1D(name, title, len(labels), 0, len(labels))
    for i, (label, count) in enumerate(zip(labels, counts), start=1):
        hist.GetXaxis().SetBinLabel(i, label)
        hist.SetBinContent(i, count)
    return hist


def crossing_histograms(data, name):
    hists = []
    # Fine wrong-side depth histogram: bin 0 holds depth 0, the last bin the overflow.
    edges = data["depth_log_histogram"]["low_edges_ktol"][1:]
    counts = data["depth_log_histogram"]["counts"]
    upper = edges[-1] * (edges[-1] / edges[-2])
    hist = ROOT.TH1D(f"crossing_depth_{name}",
                     f"{name}: {data['crossings']} GPU crossings, {counts[0]} at depth 0 or below {edges[0]:g};"
                     "wrong-side landing depth [kTolerance];landings",
                     len(edges), array.array("d", edges + [upper]))
    for i, count in enumerate(counts[1:-1], start=1):
        hist.SetBinContent(i, count)
    hist.SetBinContent(len(edges) + 1, counts[-1])
    hists.append((hist, ROOT.kAzure - 3, dict(logx=True, lines=[(1.0, ROOT.kRed + 1)])))

    off = data["off_band"]
    hists.append((categorical(
        f"crossing_off_band_{name}", f"{name}: {off['total']} off-band landings;;landings",
        ["inside a volume being left", "outside a volume being entered", "off every crossed surface",
         "proven overlap", "after a stuck-track push", "zero-step loop crossings (in band)",
         "state mismatch at step end", "state mismatch at step start"],
        [off["inside_left"], off["outside_entered"], off["off_surface"], off["proven_overlap"], off["stuck_push"],
         data.get("zero_step_loop", 0), data.get("state_mismatch_end", 0), data.get("state_mismatch_start", 0)]),
        ROOT.kRed - 4, dict(categorical=True)))

    bounce = data["bounce_histogram"]
    low = bounce["n_dot_d_low_edges"]
    hist = ROOT.TH1D(f"crossing_bounce_{name}",
                     f"{name}: {data['bounce_back']} bounce-backs;normal at the crossed surface . final direction;"
                     "crossings", len(low), low[0], 0.0)
    for i, count in enumerate(bounce["counts"], start=1):
        hist.SetBinContent(i, count)
    hists.append((hist, ROOT.kMagenta - 3, {}))
    return hists


def handoff_histograms(data, name):
    reloc, band = data["relocation"], data["tolerance_band"]
    return [(categorical(
        f"handoff_{name}", f"{name}: {data['exits']} tracks leaving the GPU regions;;tracks",
        ["confirmed", "direction reversed", "coincident surface", "crossed by handoff push", "unexplained",
         "unexplained, proven overlap", "band errors", "band errors, proven overlap"],
        [reloc["confirmed"], reloc["direction_reversed"], reloc["coincident_surface"],
         reloc["crossed_by_handoff_push"], reloc["unexplained"], reloc["unexplained_proven_overlap"], band["errors"],
         band["proven_overlap"]]), ROOT.kGreen + 2, dict(categorical=True))]


def main():
    args = parse_args()
    name = args.name or os.path.basename(args.prefix)

    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetTitleFontSize(0.035)
    ROOT.gStyle.SetPaintTextFormat(".0f")
    os.makedirs(args.output_dir, exist_ok=True)

    hists = []
    crossings = load(args.prefix + "_crossings.json")
    if crossings:
        hists += crossing_histograms(crossings, name)
    handoff = load(args.prefix + "_handoff.json")
    if handoff:
        hists += handoff_histograms(handoff, name)
    if not hists:
        raise SystemExit(f"No geometry validation output found for prefix {args.prefix}")

    root_file = ROOT.TFile.Open(os.path.join(args.output_dir, name + "_geometry_validation.root"), "RECREATE")
    keep = []
    for hist, color, options in hists:
        hist.Write()
        keep += draw(args.output_dir, hist, color, **options)
    root_file.Close()


if __name__ == "__main__":
    main()
