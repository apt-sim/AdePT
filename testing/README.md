<!--
SPDX-FileCopyrightText: 2023 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Testing

The purpose of these scripts is to automate testing and provide an easy way to define new tests.

The tests are defined in JSON files, they are parsed by a Python script which then executes the applications with different configurations, and generates plots from the output data.

There are two types of test:
  * Benchmarks: They record the time taken by different parts of the simulation, like the time spent simulating EM and Hadronic physics.
  * Validations: They record the energy deposition in sensitive volumes.

The data provided by each type of test depends on the specific implementation in the application we are testing.

## Requirements

The only requirement to be able to run these tests is to compile AdePT with the cmake option ```TEST=ON```

## Usage

In order to run a test using the script:

```python3 run_test.py configuration_file```

It will run all the tests defined in the configuration file and generate the plots.

The raw data from the tests will be saved in the specified directories, in order to generate only the plots if we already have the data:

```python3 make_plots.py configuration_file```

### Configuration file structure

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

            // A list of plots to generate from the results.
            // Each plot defines an output file and the script used to generate it.
            "plots" : 
            [
                {
                    // Output file for the plot
                    "output_file" : "test_1_histogram", 
                    // Plotting script to use
                    "executable" : "plot_histogram.py" 
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