#!/bin/bash

parent_dir="./baselines"

for subdir in "$parent_dir"/*/; do
    if [ -d "$subdir" ]; then
        echo "Processing: $subdir"

        rm -fv "$subdir"slurm-*.out
        rm -fv "$subdir"nuts.py
        rm -fv "$subdir"run-nuts.sh
    fi
done

echo "Finishing Job"
echo 0
