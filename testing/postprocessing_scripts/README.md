<!--
SPDX-FileCopyrightText: 2023 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Postprocessing Scripts

## Contents

### Transform univariate test data

This script is used to generate a single output file for univariate tests, where we do several runs of multiple events, changing the value of a parameter between runs. The output file contains the mean value across all events of each run for each measurement.

For example, in a benchmark that measures:

* Total event time
* GPU region time
* Non EM time

The output of each run would have this format, with one line per event:

|     Event    |     ECAL    |     Non EM    |
|:------------:|:-----------:|:-------------:|
| Time Event 1 | Time ECAL 1 | Time Non EM 1 |
| Time Event 2 | Time ECAL 2 | Time Non EM 2 |
| Time Event 3 | Time ECAL 3 | Time Non EM 3 |
|      ...     |     ...     |      ...      |

This script would then take the output from multiple runs, and a file with a list of labels, and generate:

|        | Label 1           | Label 2           | Label 3           | ... |
|--------|-------------------|-------------------|-------------------|-----|
|  Event |  Event mean run 1 |  Event mean run 2 |  Event mean run 3 | ... |
|  ECAL  |  ECAL mean run 1  |  ECAL mean run 2  |  ECAL mean run 3  | ... |
| Non EM | Non EM mean run 1 | Non EM mean run 2 | Non EM mean run 3 | ... |

The results from this postprocessing are meant to be plotted using the ```plot_univariate_test_data.py``` script, which takes:

* An Output file for the plot
* A label for the X axis
* A label for the Y axis
* Two files with postprocessed data, A and B

It will then plot the speedup between files A and B for each value of the variable.

