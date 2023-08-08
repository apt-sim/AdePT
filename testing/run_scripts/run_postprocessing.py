# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import os

#Uses the info in the JSON configuration file to run the postprocessing associated to the tests

def run_postprocessing(config, test_id):
    results_dir = config["results_dir"]
    postprocessing_scripts_dir = config["postprocessing_scripts_dir"]
    postprocessing_dir = config["postprocessing_dir"]

    if not os.path.exists(postprocessing_dir):
        os.mkdir(postprocessing_dir)

    test = config["tests"][test_id]

    for process_id in range(len(test["postprocessing"])):
        process = test["postprocessing"][process_id]

        print(test["postprocessing"][process_id]["output_file"])

        run_command = []
        if "arguments" in process:
            run_command = ["python3", 
                        postprocessing_scripts_dir + process["executable"], 
                        postprocessing_dir + process["output_file"],
                        process["arguments"],
                        *[results_dir + i + 
                        ".csv" for i in [j["output_file"] for j in test["runs"]]]]
        else:
            run_command = ["python3", 
                        postprocessing_scripts_dir + process["executable"], 
                        postprocessing_dir + process["output_file"],
                        *[results_dir + i + 
                        ".csv" for i in [j["output_file"] for j in test["runs"]]]]

        completed_process = subprocess.run(run_command, capture_output=True, text=True)
        
        if(completed_process.stderr != None and len(completed_process.stderr) != 0):
            print("An error occurred in postprocessing: ")
            print(completed_process.stderr)

if __name__ == '__main__':
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

    for test in range(len(config["tests"])):
        run_postprocessing(config, test)
        