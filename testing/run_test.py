# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import json
from string import Template
import subprocess
import sys
import os

if len(sys.argv) == 2:
    config_filename = sys.argv[1]
    try:
        config_file = open(config_filename)
    except:
        print(str(config_filename) + " not found in the current working directory")
        exit()
else:
    print("Usage: python3 run_test.py configuration_file")
    exit()

config = json.load(config_file)
macro_template_file = open('macro_template')
macro_template = Template(macro_template_file.read())

bin_dir = config["bin_dir"]
results_dir = config["results_dir"]
plotting_scripts_dir = config["plotting_scripts_dir"]
plots_dir = config["plots_dir"]

if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)

#Run all tests found in the configuration file
for test_id in range(len(config["tests"])):
    test = config["tests"][test_id]

    for run_id in range(len(test["runs"])):
        run = test["runs"][run_id]
        ui_dir = run["ui_dir"]

        print("Running test: " + test["name"] + ", Run: " + 
              run["name"] + " (" + str(run_id+1) + "/" + str(len(test["runs"])) + 
              ")" + 20*" ", end='\r')

        #Add a line to activate or deactivate AdePT
        if(run["use_adept"]):
            run["configuration"]["use_adept"] = "/param/ActivateModel AdePT"
        else:
            run["configuration"]["use_adept"] = "/param/InActivateModel AdePT"

        #If using the random gun, add lines for each particle and for the angles
        random_gun_configuration = ""
        if run["configuration"]["randomize_gun"]:
            particles = run["random_gun_configuration"]["particles"]
            angles = run["random_gun_configuration"]["angles"]

            #For each particle add the type, and their weight and energy if defined
            for particle in particles:
                random_gun_configuration += "/" + ui_dir + "/gun/addParticle " + particle
                if "weight" in particles[particle]:
                    random_gun_configuration += " weight " + str(particles[particle].get("weight"))
                if "energy" in particles[particle]:
                    random_gun_configuration += " energy " + str(particles[particle].get("energy"))
                random_gun_configuration += "\n"
            for angle in angles:
                random_gun_configuration += ' '.join(["/" + ui_dir + "/gun/" + angle, str(angles[angle]), "\n"])

        #Add the new lines to the configuration that will be sent to the macro
        run["configuration"]["random_gun_configuration"] = random_gun_configuration

        #Create a macro with the specified settings and save it to a file
        macro = macro_template
        macro = macro.substitute(run["configuration"])
        temp_macro = open("temp_macro", "w")
        temp_macro.write(macro)
        temp_macro.close()

        #Choose the proper option to select the data we extract
        test_option = ""
        if test["type"] == "benchmark":
            test_option = "--do_benchmark"
        elif test["type"] == "validation":
            test_option = "--do_validation"

        #Run the simulation
        completed_process = subprocess.run([bin_dir + run["executable"], 
                                        "-m", "temp_macro",
                                        "--output_dir", results_dir,
                                        "--output_file", run["output_file"],
                                        test_option], capture_output=True, text=True)

        if(completed_process.stderr != None and len(completed_process.stderr) != 0):
            print("An error occurred in run: " + run["name"])
            print(completed_process.stderr)
        
        os.remove("temp_macro")
    print()
    
    #After all the runs in this test have finished generate the plots configured for it
    for plot_id in range(len(test["plots"])):
        plot = test["plots"][plot_id]
        completed_process = subprocess.run(["python3", 
                        plotting_scripts_dir + plot["executable"], 
                        plots_dir + plot["output_file"],
                        *[results_dir + i + 
                          ".csv" for i in [j["output_file"] for j in test["runs"]]]], 
                        capture_output=True, text=True)
        
        if(completed_process.stderr != None and len(completed_process.stderr) != 0):
            print("An error occurred plotting the results: ")
            print(completed_process.stderr)

