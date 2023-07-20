# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import os

#Uses the info in the JSON configuration file to generate just the plots associated to the tests

if len(sys.argv) == 2:
    config_filename = sys.argv[1]
    try:
        config_file = open(config_filename)
    except:
        print(str(config_filename) + " not found in the current working directory")
        exit()
else:
    print("Usage: python3 make_plots.py configuration_file")
    exit()

config = json.load(config_file)

results_dir = config["results_dir"]
plotting_scripts_dir = config["plotting_scripts_dir"]
plots_dir = config["plots_dir"]

if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)

for test_id in range(len(config["tests"])):
    test = config["tests"][test_id]

    for plot_id in range(len(test["plots"])):
        plot = test["plots"][plot_id]

        print(test["plots"][plot_id]["output_file"])

        completed_process = subprocess.run(["python3", 
                        plotting_scripts_dir + plot["executable"], 
                        plots_dir + plot["output_file"],
                        *[results_dir + i + 
                          ".csv" for i in [j["output_file"] for j in test["runs"]]]], 
                        capture_output=True, text=True)
        if(completed_process.stderr != None and len(completed_process.stderr) != 0):
            print("An error occurred plotting the results: ")
            print(completed_process.stderr)
        
