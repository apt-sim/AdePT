<!--
SPDX-FileCopyrightText: 2023 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Testing

The purpose of these scripts is to automate testing and provide an easy way to define new tests.

The tests are defined in JSON files, they are parsed by a Python script which then executes the applications with different configurations, generates plots, and does postprocessing on the output data.

There are two types of test:
  * **Benchmarks:** They record the time taken by different parts of the simulation, like the time spent simulating EM and Hadronic physics.
  * **Validations:** They record the energy deposition in sensitive volumes.

The data provided by each type of test depends on the specific implementation in the application we are testing.

## Requirements

The only requirement to be able to run these tests is to compile AdePT with the cmake option ```TEST=ON```

## Contents

### Run Scripts

Scripts used to run the tests.

These scripts take JSON files containing test configurations as input and perform the specified runs.

A basic test has three components:

* Runs: A list of the runs to be executed. Each run contains the full configuration for the Geant4 macro file.
* Postprocessing scripts: A list of postprocessing scripts applied to the output from the runs. 
* Plotting scripts: A list of plotting scripts applied to the output from the runs. 

### Postprocessing Scripts

Scripts used for postprocessing of the output from the runs.

These scripts should take:

* An Output file for the results
* A list of optional arguments
* A list of input files

### Plotting Scripts

Scripts used for plotting the output from the runs.

These scripts should take:

* An Output file for the plot
* A label for the X axis
* A label for the Y axis
* A list of input files

### Templates

Geant4 macro templates to be used in the tests. The substitution is done using the ```string.Template.substitute``` Python3 method:

https://docs.python.org/3/library/string.html#string.Template.substitute

In order to implement a template:

* Every parameter that needs to be changed dynamically should be defined with the syntax ```$variable```
* In the JSON configuration file, the "configuration" section for each run needs to include all the variables defined in the template

## Usage

In order to run a test using the script:

```python3 run_test.py configuration_file```

It will run all the tests defined in the configuration file, and run the scripts configured for plotting and postprocessing of the results.

The raw data from the tests will be saved in the specified directories, in order to generate only the plots if we already have the data:

```python3 make_plots.py configuration_file```

In order to run only the postprocessing;

```python3 run_postprocessing.py configuration_file```

### Usage example

In order to run a benchmark with a basic configuration:

```python3 run_scripts/run_test.py test_configurations/benchmark_basic.json```

## Configuration file structure

Configuration files must be in JSON format.

Usage example:

``` jsonc
{
    // Directory containing the AdePT binaries
    "bin_dir" : "../adept-build/BuildProducts/bin/", 
    // Output directory for the raw results
    "results_dir" : "results/", 
    // Directory containing the python plotting scripts
    "plotting_scripts_dir" : "plotting_scripts/", 
    // Output directory for the plots
    "plots_dir" : "plots/", 
    
    // An array of the tests we want to perform, a test contains a list of runs and a list of plots.
    "tests": 
    [
        {
            // Identifies the test
            "name" : "Test 1", 
            // There are two types of test
            // - validation: outputs the energy deposition in the sensitive volumes
            // - benchmark: outputs the time spent inside and out of the GPU region
            "type" : "validation|benchmark", 

            // A list of plots to generate from the results
            // For each plot we need to define:
            // * An output file
            // * The script used to generate it
            // * A label for the X axis
            // * A label for the Y axis
            "plots" : 
            [
                {
                    "output_file" : "test_1_plot", 
                    "executable" : "plot_bar_chart.py" ,
                    "x_label" : "x label",
                    "y_label" : "y label"
                }
            ],

            // A list of postprocessing scripts to apply to the results
            // For each script we need to define:
            // * An output file
            // * Optionally, a list of arguments for the script
            // * The script to run
            "postprocessing" :
            [
                {
                    "output_file" : "test_1_postprocessing",
                    "arguments" : "arg1 arg2 arg3",
                    "executable" : "transform_univariate_test_data.py"
                }
            ],

            // A list of configurations we want to test
            "runs": 
            [
                {
                    // Identifies the run
                    "name" : "Run 1", 
                    // The application to run
                    "executable" : "example", 
                    // Output file for the raw results
                    "output_file" : "example_adept", 
                    // Name of the UI directory used by the application, for the macro commands
                    "ui_dir" : "example", 
                    // Whether to use AdePT
                    "use_adept" : true, 

                    // The configuration for the macro file
                    "configuration" : 
                    {
                        // Number of worker threads
                        "num_threads" : 1, 
                        // Input GDML file
                        "gdml_file" : "../adept-build/cms2018.gdml", 
                        // Number of particles that need to enter the buffer to trigger AdePT simulation
                        "adept_threshold" : 2000, 
                        // Millions of track slots to pre-allocate
                        "adept_million_track_slots" : 10,
                        // Strength and direction of the magnetic field
                        "magnetic_field" : "0 0 0 tesla",
                        // Type of particle to shoot
                        "particle_type" : "e-", 
                        // Energy of the particles
                        "gun_energy" : "10 GeV",  
                        // Number of particles to shoot
                        "num_particles" : 2000, 
                        // If true the random gun configuration is needed as well
                        "randomize_gun" : true,
                        // Seed for the random number generator
                        "random_seed" : 1,
                        // Number of events to simulate
                        "num_events" : 5 
                    },
                    
                    // Defines a list of particles to shoot, and optionally their weights and energies
                    // Defines the range in which to shoot in terms of Theta and Phi
                    "random_gun_configuration" :
                    {
                        "particles" :
                        {
                            "e-" : {"weight" : 0.4, "energy" : "15 GeV"}
                        },
                        "angles" :
                        {
                            "minPhi" : 0,
                            "maxPhi" : 360,
                            "minTheta" : 10,
                            "maxTheta" : 170
                        }
                    }
                }
            ]
        }
    ]
}
```