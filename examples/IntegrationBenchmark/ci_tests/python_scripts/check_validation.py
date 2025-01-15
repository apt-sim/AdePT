#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CERN
# SPDX-License-Identifier: Apache-2.0
import argparse
import numpy as np

import sys

def compare_csv(file1, file2, n1, n2, tol=0.01, plot_file=None):
    """
    Compares two CSV files column by column and checks for relative error exceeding the given tolerance.

    :param file1: Path to the first CSV file
    :param file2: Path to the second CSV file
    :param tol: Tolerance for relative error (in percentage)
    """
    # Load the CSV files
    try:
        data1 = np.loadtxt(file1, delimiter=",", skiprows=1)
        data2 = np.loadtxt(file2, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # Sum rows to combine events into total energy deposition per column
    sum1 = np.sum(data1, axis=0)
    sum2 = np.sum(data2, axis=0)

    # drop the first index as it doesn't correspond to energy deposition
    sum1 = sum1[1:]
    sum2 = sum2[1:] 
    # Sum every two consecutive values two sum lead and liquid argon layer energy deposition into one combined layer
    sum1 = sum1[::2] + sum1[1::2] 
    sum2 = sum2[::2] + sum2[1::2] 
    # ignore output past first 50 layers
    sum1 = sum1[:50]
    sum2 = sum2[:50]

    sum1_normalized = sum1 / n1
    sum2_normalized = sum2 / n2


    # Compute relative error in percentage
    relative_error = np.abs((sum1_normalized - sum2_normalized) / sum2_normalized) * 100

    # Check for columns exceeding the tolerance
    failed_layers = []
    for i, err in enumerate(relative_error):
        if err > tol * 100:
            failed_layers.append((i + 1, err, sum1_normalized[i], sum2_normalized[i]))  # layer number (1-based), error, file1 value, file2 value


    if plot_file:
        from matplotlib import pyplot as plt

        # Generate a plot
        layers = np.arange(1, 51)
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # Mean energy deposit plot
        ax[0].plot(layers, sum2_normalized, 'o-', label="G4 + HepEm", markersize=5, markerfacecolor="none", markeredgewidth=1, markeredgecolor="C1", color="C1")
        ax[0].plot(layers, sum1_normalized, 'o', label="AdePT + HepEm", markersize=4.5, markerfacecolor="C0", markeredgewidth=0, color="C0")
        ax[0].text(7, 0.9 * max(sum2_normalized), f"$N_1 = {n1}, N_2 = {n2}$", fontsize=10)
        ax[0].set_ylabel("Mean energy deposit [MeV]")
        ax[0].legend(frameon=False)

        # Relative error plot
        ax[1].bar(layers, (sum1_normalized - sum2_normalized) / sum2_normalized * 100, label="Relative error")
        ax[1].set_ylim(-1, max(1, tol + 1))
        ax[1].set_xlabel("# layer")
        ax[1].set_ylabel("err in %")

        plt.tight_layout()

        # Save the plot
        plt.savefig(plot_file)
        print(f"Plot saved to {plot_file}")


    # Print results
    if failed_layers:
        print(f"The physics results are not valid. Relative errors exceed {100*tol}% in the following layers:")
        for layer, err, val1, val2 in failed_layers:
            print(f"Layer {layer}: Relative Error = {err:.6f}%, File1 = {val1:.6f}, File2 = {val2:.6f}")
        sys.exit(1)
    else:
        print(f"The physics results are valid. All layers have relative errors within {100*tol}%.")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare energy deposition between two datasets and check relative error between them.")
    parser.add_argument("--file1", required=True, help="Path to the first CSV file.")
    parser.add_argument("--file2", required=True, help="Path to the second CSV file.")
    parser.add_argument("--n1", required=True, type=float, help="Number of primary particles used in the first dataset.")
    parser.add_argument("--n2", required=True, type=float, help="Number of primary particles used in the first dataset.")
    parser.add_argument("--tol", type=float, default=0.01, help="Tolerance for relative error (default: 0.01).")
    parser.add_argument("--plot_file", default=None, help="Optional path to save the comparison plot as an image.")

    # Parse arguments
    args = parser.parse_args()

    # Call the comparison function with parsed arguments
    compare_csv(args.file1, args.file2, args.n1, args.n2, args.tol, args.plot_file)

