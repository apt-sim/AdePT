# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import json
from string import Template
import subprocess
import sys
import os
from make_plots import *
from run_postprocessing import *

if len(sys.argv) < 2:
    print("Usage: python3 run_test.py configuration_file [configuration_file_2 ...]")
    exit()

for config_filename in sys.argv[1:]:
    try:
        config_file = open(config_filename)
    except:
        print("Configuration file: " + str(config_filename) + " not found")
        continue

    config = json.load(config_file)
    macro_template_file = open(config["templates_dir"] + config["macro_template"])
    macro_template = Template(macro_template_file.read())

    bin_dir = config["bin_dir"]
    results_dir = config["results_dir"]
    plotting_scripts_dir = config["plotting_scripts_dir"]
    plots_dir = config["plots_dir"]
    postprocessing_scripts_dir = config["postprocessing_scripts_dir"]
    postprocessing_dir = config["postprocessing_dir"]

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    if not os.path.exists(postprocessing_dir):
        os.mkdir(postprocessing_dir)

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
            run["configuration"]["ui_dir"] = ui_dir

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

            #If the output file already exists, overwrite it
            if os.path.exists(results_dir + "/" + run["output_file"] + ".csv"):
                os.remove(results_dir + "/" + run["output_file"] + ".csv")

            #Run the simulation
            completed_process = subprocess.run([bin_dir + run["executable"], 
                                            "-m", "temp_macro",
                                            "--output_dir", results_dir,
                                            "--output_file", run["output_file"],
                                            test_option], capture_output=True, text=True)

            if(completed_process.stderr != None and len(completed_process.stderr) != 0):
                print("\nAn error occurred in run: " + run["name"])
                print(completed_process.stderr)
            
            os.remove("temp_macro")
        print()
        
        #After all the runs in this test have finished generate the plots configured for it
        print("Generating plots:\n")
        make_plots(config, test_id)
        
        #Run the postprocessing
        print("Running postprocessing:\n")
        run_postprocessing(config, test_id)

