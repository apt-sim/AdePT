# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler

if len(sys.argv) < 6:
	print("Usage: python3 plot_bar_chart.py output_file x_label y_label max_values data.csv [data_2.csv data_3.csv ...]")
	exit()

output_file = sys.argv[1]
x_label = sys.argv[2]
y_label = sys.argv[3]
# max_values limits the number of bars we print to the x largest values, 0 for no limit
max_values = int(sys.argv[4])
data_files = sys.argv[5:]


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#Uncomment this line for custom color cycle
#plt.rc('axes', prop_cycle=(cycler('color', ['c', 'orange', 'r', 'g', 'yellow', 'violet'])))

plt.figure(figsize=(20, 12))

#Adjust the width based on the number of bars we need to plot, leaving space on the sides
width=0.6/(len(sys.argv)-2)

for i in range(len(data_files)):
	file = data_files[i]
	data = pd.read_csv(file)

	#Sort the columns in ascending order
	data = data.reindex(data.mean().sort_values().index, axis=1)
	
	#Get the mean of each timing
	means = data.mean()
	#Get the standard error for each timing.
	errors = [np.std(data[column]) for column in data.columns]
	
	if(max_values):
		x = np.arange(max_values)
	else:
		x = np.arange(len(data.columns))
	
	#Draw grid below other figures
	plt.gca().set_axisbelow(True)
	plt.grid(True, axis='y', color='black', linestyle='dotted')
	
	#Plot the data in a bar chart
	plt.bar(x=x+((i-len(data_files)//2)*width), height=means[-max_values:], yerr=errors[-max_values:], width=width, label=file)
	#plt.errorbar(x=x+((i-len(data_files)//2)*width), y=means[-max_values:] , yerr=errors[-max_values:], fmt='none', color="black")
	plt.xticks(x, data.columns[-max_values:])
	plt.ylabel(y_label)
	plt.xlabel(x_label)
	#plt.yscale("log")
	plt.legend()

#plt.show()
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
