# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 4:
	print("Usage: python3 plot_proportions_test.py output_file x_labels_file [data.csv data_2.csv ...]")
	exit()
	
output_file = sys.argv[1]
x_labels_file = sys.argv[2]
data_files = sys.argv[3:]

output = pd.DataFrame()
#This has to follow the order in which the runs are defined in the configuration
x_labels = open(x_labels_file).read().strip().split(",")

for i in range(len(data_files)):
    file = data_files[i]
    data = pd.read_csv(file)
    means = data.mean()
    output[x_labels[i]] = means

output.to_csv(output_file + ".csv")
