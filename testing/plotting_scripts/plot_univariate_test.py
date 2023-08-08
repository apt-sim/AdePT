# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler
import matplotlib.ticker as ticker

if len(sys.argv) < 4:
	print("Usage: python3 plot_proportions_test.py output_file x_label y_label data.csv data_2.csv")
	exit()

output_file = sys.argv[1]
x_label = sys.argv[2]
y_label = sys.argv[3]
data_files = sys.argv[4:]

#Uncomment this line for custom color cycle
#plt.rc('axes', prop_cycle=(cycler('color', ['c', 'orange', 'r', 'g', 'yellow', 'violet'])))
markercycle = cycler(marker=['s', '^'])
colorcycle = cycler(color=['r', 'g', 'b'])

width=0.2

fig,axs = plt.subplots(2, 2)

fig.set_figwidth(30)
fig.set_figheight(15)

axs = axs.flatten()

for ax in axs:
	ax.set_prop_cycle(markercycle * colorcycle)
	#Draw grid below other figures
	ax.set_axisbelow(True)
	ax.grid(True, axis='y', color='black', linestyle='dotted')

#Set the same y-scale for the axes showing the ratio between values
ratio_axes = axs[1:]
for ax_idx in range(len(ratio_axes)):
	ratio_axes[ax_idx].sharey(ratio_axes[ax_idx-1])

f1 = data_files[0]
d1 = pd.read_csv(f1)
f2 = data_files[1]
d2 = pd.read_csv(f2)

x = np.arange(len(d1.columns))

#Plot all data in the first axis
for i in range(len(d1.values)):
	#Plot the data
	axs[0].plot(d1.columns[1:], d1.values[i][1:], label=f1[f1.rindex("/"):] + " " + d1.values[i][0], linewidth=1)
for i in range(len(d2.values)):
	axs[0].plot(d2.columns[1:], d2.values[i][1:], label=f2[f2.rindex("/"):] + " " + d2.values[i][0], linewidth=1)
#plt.xticks(x, data.columns)
axs[0].set(ylabel="Time (s)")
axs[0].set(xlabel=x_label)
axs[0].legend()

#For the other axes, plot the ratio between each value
for row_idx in range(len(d1.values)):
	row1 = d1.values[row_idx]
	row2 = d2.values[row_idx]
	ax = axs[row_idx+1]
	ax.plot(d1.columns[1:], row1[1:]/row2[1:], label="Speedup " + row1[0])
	ax.set_ylim(bottom=0)
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.set_title("Speedup " + row1[0])
	ax.set(ylabel="Speedup")
	ax.set(xlabel=x_label)
	ax.legend()


#plt.show()
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
