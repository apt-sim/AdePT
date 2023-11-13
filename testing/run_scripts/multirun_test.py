# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import shutil

from run_test import *

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 multirun_test.py n_runs results_dir configuration_filename")
        exit()

    n_runs = int(sys.argv[1])
    results_dir = sys.argv[2]
    configuration_filename = sys.argv[3]

    # Run the test n times
    for i in range(n_runs):
        # Run the test
        run_test(i, [configuration_filename])

        # Copy the output to the specified results directory
        try:
            config_file = open(configuration_filename)
        except:
            print("Configuration file: " + str(configuration_filename) + " not found")
            continue
        config = json.load(config_file)

        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        
        for test in config["tests"]:
            for run in test["runs"]:
                print("Copying " + config["results_dir"] + "/" + run["output_file"] + ".csv")
                print("Copying " + config["results_dir"] + "/" + run["output_file"] + "_global.csv")
                if not os.path.exists(results_dir + "/" + str(i)):
                    os.mkdir(results_dir + "/" + str(i))
                shutil.copy(config["results_dir"] + "/" + run["output_file"] + ".csv", results_dir + "/" + str(i))
                shutil.copy(config["results_dir"] + "/" + run["output_file"] + "_global.csv", results_dir + "/" + str(i))
