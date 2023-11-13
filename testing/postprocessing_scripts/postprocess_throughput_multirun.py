# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import pandas as pd
import numpy as np
import os

# Postprocess multirun results and call plot points with the appropiate data

if len(sys.argv) < 2:
    print("Usage: python3 multirun_test.py results_dir configuration_filename")
    exit()

results_dir = sys.argv[1]
configuration_filename = sys.argv[2]

try:
    config_file = open(configuration_filename)
except:
    print("Configuration file: " + str(configuration_filename) + " not found")
config = json.load(config_file)

n_runs = len(os.listdir(results_dir))

# For each different run, get the average of the global parameters it recorded
for test in config["tests"]:
    thread_labels = ["1", "2", "4", "8", "16"]#, "32", "64", "96"]
    results = pd.DataFrame(columns=thread_labels)
    for run in test["runs"]:
        time_means = []
        particle_means = []
        
        for i in range(n_runs):
            #Load each file and sum up the results per column
            data = pd.read_csv(results_dir + "/" + str(i) + "/" + run["output_file"] + "_global.csv")
            data.columns = data.columns.str.strip()
            time_means.append(data["Totaltime"].mean())
            particle_means.append(data["NumParticles"].mean())
        
        #Only works for tests that define runs with the threads scaling in the same order as declared here
        run_idx = test["runs"].index(run)
        #In case we don't want to use all the runs from a test
        if run_idx > len(thread_labels)-1:
            break
        results[thread_labels[run_idx]] = np.array(particle_means) / np.array(time_means)

    print(results)

    print(test["name"])

    if "AdePT" in test["name"]:
        if "with field" in test["name"]:
            results.to_csv("AdePT_Throughput_field", index=False)
        else:
            results.to_csv("AdePT_Throughput", index=False)

    else:
        if "with field" in test["name"]:
            results.to_csv("Geant4_Throughput_field", index=False)
        else:
            results.to_csv("Geant4_Throughput", index=False)






