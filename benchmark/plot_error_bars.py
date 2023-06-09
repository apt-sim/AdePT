# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler

if len(sys.argv) < 2:
	print("Usage: python3 plot_error_bars.py [data.csv] [data_2.csv data_3.csv ...]")
	exit()

plt.rc('axes', prop_cycle=(cycler('color', ['c', 'orange', 'b', 'g'])))

for i in range(len(sys.argv[1:])):
	file = sys.argv[i+1]
	data = pd.read_csv(file)
	#Sort the columns in ascending order
	data = data.reindex(data.mean().sort_values().index, axis=1)
	print(data)
	#Get the mean of each timing
	means = data.mean()
	#Get the standard error for each timing.
	errors = [np.std(data[column]) for column in data.columns]

	width=0.2
	x = np.arange(len(data.columns))
	#Draw grid below other figures
	plt.gca().set_axisbelow(True)
	plt.grid(True, axis='y', color='black', linestyle='dotted')
	#Plot the data in a bar chart
	plt.bar(x=x+((i-len(sys.argv[1:])//2)*width), height=means, yerr=errors, width=width, label=file)
	plt.xticks(x, data.columns)
	plt.ylabel("Time (s)")
	plt.xlabel("Event")
	plt.legend()
plt.show()