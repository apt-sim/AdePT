# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler

if len(sys.argv) < 3:
	print("Usage: python3 plot_points.py output_file data.csv [data_2.csv data_3.csv ...]")
	exit()

output_file = sys.argv[1]
data_files = sys.argv[2:]

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#Uncomment this line for custom color cycle
#plt.rc('axes', prop_cycle=(cycler('color', ['c', 'orange', 'r', 'g', 'yellow', 'violet'])))

width=0.2

plt.figure(figsize=(18, 10))

for i in range(len(data_files)):
	file = data_files[i]
	data = pd.read_csv(file)

	#Sort the columns in ascending order
	data = data.reindex(data.mean().sort_values().index, axis=1)
	
	#Get the mean of each timing
	means = data.mean()
	#Get the standard error for each timing.
	errors = [np.std(data[column]) for column in data.columns]

	x = np.arange(len(data.columns))
	
	#Draw grid below other figures
	plt.gca().set_axisbelow(True)
	plt.grid(True, axis='y', color='black', linestyle='dotted')
	
	#Plot the data
	plt.errorbar(x=x+((i-len(sys.argv[1:])//2)*width), y=means, yerr=errors, label=file, linewidth=0, marker="s", elinewidth=1)
	plt.xticks(x, data.columns)
	plt.ylabel("Time (s)")
	plt.legend()

#plt.show()
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
