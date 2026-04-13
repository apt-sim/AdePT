#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2026 CERN
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import sys

# FIXME: G4 track IDs are not reproducible between runs, and the current
# nuclear-process callback ordering will not assign the parent ID to the created secondaries, but treat them as primaries with randomly varying Geant4 IDs, hence changing the primary ancestor histograms. Until that core ordering issue is fixed,
# the primary_ancestor_population histogram is expected to be unstable and must
# be excluded from the exact ROOT comparison.
IGNORED_HISTOGRAMS = {"primary_ancestor_population"}


def load_root():
    try:
        import ROOT  # type: ignore
    except Exception as exc:
        print(f"Failed to import ROOT: {exc}")
        sys.exit(1)

    ROOT.gROOT.SetBatch(True)
    return ROOT


def list_histograms(root_file):
    # Only the TH1 payload is part of the comparison inventory. Sidecar metadata
    # objects are loaded separately when needed.
    histograms = {}
    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()
        if not obj.InheritsFrom("TH1"):
            continue
        if key.GetName() in IGNORED_HISTOGRAMS:
            continue
        histograms[key.GetName()] = obj
    return histograms


def load_value_metadata(root_file, histogram_name):
    # Continuous observables are stored as count histograms plus a sidecar
    # TObjString that lists the exact floating-point values in bin order.
    metadata = root_file.Get(f"{histogram_name}__values")
    if not metadata:
        return None
    return metadata.GetString().Data()


def load_label_metadata(root_file, histogram_name):
    # Categorical labels are stored in a sidecar TObjString so the comparison
    # does not depend on ROOT TH1 axis-label internals.
    metadata = root_file.Get(f"{histogram_name}__labels")
    if not metadata:
        return None
    return metadata.GetString().Data()


def compare_single_histogram(name, hist1, hist2, root1, root2, abs_tol, rel_tol):
    if hist1.GetNbinsX() != hist2.GetNbinsX():
        return f"Histogram '{name}' differs: bin count {hist1.GetNbinsX()} != {hist2.GetNbinsX()}"

    metadata1 = load_value_metadata(root1, name)
    metadata2 = load_value_metadata(root2, name)
    label_metadata1 = load_label_metadata(root1, name)
    label_metadata2 = load_label_metadata(root2, name)
    values1 = metadata1.splitlines() if metadata1 is not None else None
    values2 = metadata2.splitlines() if metadata2 is not None else None
    labels1 = label_metadata1.splitlines() if label_metadata1 is not None else None
    labels2 = label_metadata2.splitlines() if label_metadata2 is not None else None

    # For exact-value histograms we first compare the metadata that defines
    # which floating-point value each bin corresponds to, and only then the
    # counts stored in those bins.
    if (metadata1 is None) != (metadata2 is None):
        return f"Histogram '{name}' differs: exact-value metadata is missing in one file."

    if metadata1 is not None and metadata1 != metadata2:
        for index, (value1, value2) in enumerate(zip(values1, values2), start=1):
            if value1 != value2:
                return f"Histogram '{name}' differs at bin {index}: value '{value1}' != '{value2}'"
        return (
            f"Histogram '{name}' differs: exact-value metadata length "
            f"{len(values1)} != {len(values2)}"
        )

    if (label_metadata1 is None) != (label_metadata2 is None):
        return f"Histogram '{name}' differs: label metadata is missing in one file."

    if label_metadata1 is not None and label_metadata1 != label_metadata2:
        for index, (label1, label2) in enumerate(zip(labels1, labels2), start=1):
            if label1 != label2:
                return f"Histogram '{name}' differs at bin {index}: label '{label1}' != '{label2}'"
        return (
            f"Histogram '{name}' differs: label metadata length "
            f"{len(labels1)} != {len(labels2)}"
        )

    for bin_index in range(1, hist1.GetNbinsX() + 1):
        if values1 is not None:
            label1 = values1[bin_index - 1]
        elif labels1 is not None:
            label1 = labels1[bin_index - 1]
        else:
            label1 = hist1.GetXaxis().GetBinLabel(bin_index)

        if values2 is not None:
            label2 = values2[bin_index - 1]
        elif labels2 is not None:
            label2 = labels2[bin_index - 1]
        else:
            label2 = hist2.GetXaxis().GetBinLabel(bin_index)

        if label1 != label2:
            return f"Histogram '{name}' differs at bin {bin_index}: label '{label1}' != '{label2}'"

        value1 = hist1.GetBinContent(bin_index)
        value2 = hist2.GetBinContent(bin_index)
        if not math.isclose(value1, value2, rel_tol=rel_tol, abs_tol=abs_tol):
            if not label1:
                label1 = f"bin={bin_index}"
            return (
                f"Histogram '{name}' differs at bin {bin_index} ('{label1}'): "
                f"file1={value1} file2={value2}"
            )

    return None


def compare_histograms(file1, file2, abs_tol, rel_tol):
    ROOT = load_root()
    root1 = ROOT.TFile.Open(file1, "READ")
    root2 = ROOT.TFile.Open(file2, "READ")

    if not root1 or root1.IsZombie():
        print(f"Failed to open ROOT file: {file1}")
        sys.exit(1)
    if not root2 or root2.IsZombie():
        print(f"Failed to open ROOT file: {file2}")
        sys.exit(1)

    histograms1 = list_histograms(root1)
    histograms2 = list_histograms(root2)

    names1 = sorted(histograms1.keys())
    names2 = sorted(histograms2.keys())
    mismatches = []
    if names1 != names2:
        only_in_file1 = sorted(set(names1) - set(names2))
        only_in_file2 = sorted(set(names2) - set(names1))
        if only_in_file1:
            mismatches.append(f"Histogram inventory differs: only in file1: {only_in_file1}")
        if only_in_file2:
            mismatches.append(f"Histogram inventory differs: only in file2: {only_in_file2}")

    common_names = sorted(set(names1) & set(names2))
    for name in common_names:
        mismatch = compare_single_histogram(name, histograms1[name], histograms2[name], root1, root2, abs_tol, rel_tol)
        if mismatch is not None:
            mismatches.append(mismatch)

    if mismatches:
        print(
            "ROOT histogram comparison found mismatches: "
            f"{len(mismatches)} failing checks across {len(common_names)} common histograms."
        )
        for mismatch in mismatches:
            print(f"- {mismatch}")
        sys.exit(1)

    print(
        "ROOT histogram outputs match within the configured tolerances: "
        f"abs_tol={abs_tol:g}, rel_tol={rel_tol:g}; "
        f"checked {len(common_names)} common histograms."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ROOT histogram outputs with small numeric tolerances.")
    parser.add_argument("--file1", required=True, help="Path to the first ROOT file.")
    parser.add_argument("--file2", required=True, help="Path to the second ROOT file.")
    parser.add_argument("--abs_tol", type=float, default=1.0e-10, help="Absolute tolerance for numeric bin contents.")
    parser.add_argument("--rel_tol", type=float, default=1.0e-14, help="Relative tolerance for numeric bin contents.")
    arguments = parser.parse_args()

    compare_histograms(arguments.file1, arguments.file2, arguments.abs_tol, arguments.rel_tol)
