#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2025 CERN
# SPDX-License-Identifier: Apache-2.0

import argparse
import pandas as pd
import sys

def compare_csv(file1, file2, tol=0.0):
    # Load the CSV files
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # Ensure the headers match
    if list(df1.columns) != list(df2.columns):
        print("Headers do not match between the two files!")
        sys.exit(1)

    # Convert tolerance to float
    tol = float(tol)

    # Ensure only numeric columns are compared
    df1 = df1.apply(pd.to_numeric, errors='coerce')
    df2 = df2.apply(pd.to_numeric, errors='coerce')

    # Drop any rows or columns with NaN values (optional, depending on use case)
    df1.dropna(inplace=True)
    df2.dropna(inplace=True)

    # Sum the rows (which are the Events) for each file
    sum1 = df1.sum(axis=0)
    sum2 = df2.sum(axis=0)

    # Compare column by column (which is volume by volume but also the other accumulators)
    differences = []
    for column in df1.columns:
        if abs(sum1[column] - sum2[column]) > tol:
            differences.append((column, sum1[column], sum2[column]))

    # Print results
    if differences:
        print("The results are not reproducible. Differences found")
        for col, val1, val2 in differences:
            print(f"Column '{col}': File1 = {val1}, File2 = {val2}")
        sys.exit(1)
    else:
        print("The results are reproducible. The files are identical after summing Events to a tolerance of " + str(tol))


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare energy deposition between two datasets and check relative error between them.")
    parser.add_argument("--file1", required=True, help="Path to the first CSV file.")
    parser.add_argument("--file2", required=True, help="Path to the second CSV file.")
    parser.add_argument("--tol", type=float, default=0., help="Tolerance for relative error (default: 0.0).")

    # Parse arguments
    args = parser.parse_args()

    # Call the comparison function with parsed arguments
    compare_csv(args.file1, args.file2, args.tol)