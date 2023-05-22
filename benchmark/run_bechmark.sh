# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0
#!/bin/bash

if [ $# -ne 3 ]
then
	echo "Usage: run_benchmark [application] [number of runs] [output file]"
	exit
fi

echo "Running benchmark for application "$1
echo "Number of runs: "$2
echo "Output file: benchmark/"$3".csv"

for i in $(seq 1 $2)
do
	echo -ne "Current run: "$i\\r
	$1 -b $3 > /dev/null
done
