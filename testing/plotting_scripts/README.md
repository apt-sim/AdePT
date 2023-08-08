<!--
SPDX-FileCopyrightText: 2023 CERN
SPDX-License-Identifier: CC-BY-4.0
-->

# Plotting Scripts

## Usage

The scripts in this directory should take as input:

* An Output file for the plot
* A label for the X axis
* A label for the Y axis
* A list of input files

## Contents

### Plot bar chart

Takes a list of run outputs in the format:

| Label 1       | Label 2       | Label 3       | ... |
|---------------|---------------|---------------|-----|
| value event 1 | value event 1 | value event 1 | ... |
| value event 2 | value event 2 | value event 2 | ... |
| value event 3 | value event 3 | value event 3 | ... |
| ...           | ...           | ...           | ... |

And produces a bar chart, which will have one bar representing the mean value per run and per label, and with error bars.

### Plot points

Takes a list of run outputs in the format:

| Label 1       | Label 2       | Label 3       | ... |
|---------------|---------------|---------------|-----|
| value event 1 | value event 1 | value event 1 | ... |
| value event 2 | value event 2 | value event 2 | ... |
| value event 3 | value event 3 | value event 3 | ... |
| ...           | ...           | ...           | ... |

And produces a chart with one point representing the mean value per run and per label, and with error bars.

### Plot ratio

Takes two run outputs in the format:

A:

| Label 1       | Label 2       | Label 3       | ... |
|---------------|---------------|---------------|-----|
| value event 1 | value event 1 | value event 1 | ... |
| value event 2 | value event 2 | value event 2 | ... |
| value event 3 | value event 3 | value event 3 | ... |
| ...           | ...           | ...           | ... |

B:

| Label 1       | Label 2       | Label 3       | ... |
|---------------|---------------|---------------|-----|
| value event 1 | value event 1 | value event 1 | ... |
| value event 2 | value event 2 | value event 2 | ... |
| value event 3 | value event 3 | value event 3 | ... |
| ...           | ...           | ...           | ... |

It then computes ```C``` as the element-wise division of ```A``` by ```B```

C:

| Label 1                 | Label 2                 | Label 3                 | ... |
|-------------------------|-------------------------|-------------------------|-----|
| value e1 A / value e1 B | value e1 A / value e1 B | value e1 A / value e1 B | ... |
| value e2 A / value e2 B | value e2 A / value e2 B | value e2 A / value e2 B | ... |
| value e3 A / value e3 B | value e3 A / value e3 B | value e3 A / value e3 B | ... |
| ...                     | ...                     | ...                     | ... |

Finally, it produces a chart with one point per column of ```C``` representing the average value of the ratio, with error bars.

### Plot univariate test

Used on postprocessed output from univariate tests. Takes two input files ```A``` and ```B``` with format:

|        | Label 1            | Label 2           | Label 3           | ... |
|--------|--------------------|-------------------|-------------------|-----|
|  Tag 1 |  Tag 1 mean run 1  |  Tag 1 mean run 2 |  Tag 1 mean run 3 | ... |
|  Tag 2 |  Tag 2 mean run 1  |  Tag 2 mean run 2 |  Tag 2 mean run 3 | ... |
|  Tag 3 | Tag 3 mean run 1   | Tag 3 mean run 2  | Tag 3 mean run 3  | ... |
|  ...   | ...                | ...               | ...               | ... |

It produces several plots, one will contain one line per Tag, with the values on the Y axis and the labels on the X axis. The rest of the plots represent the ratio between each tag, the values from file ```A``` divided by the ones in file ```B```.