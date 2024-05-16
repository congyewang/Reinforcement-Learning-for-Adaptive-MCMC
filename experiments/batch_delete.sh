#!/bin/bash


BASELINES_DIR="./baselines"


if [ ! -d "$BASELINES_DIR" ]; then
    echo "Dir $BASELINES_DIR does not exist."
    exit 1
fi

for subdir in "$BASELINES_DIR"/*; do
    if [ -d "$subdir" ]; then
        for subsubdir in "$subdir"/*; do
            if [ -d "$subsubdir" ]; then
                echo "Processing: $subsubdir"

                rm -fv "$subsubdir"/slurm-*.out
                rm -fv "$subsubdir"/mala.npy
                rm -fv "$subsubdir"/mala.py
                rm -fv "$subsubdir"/run-mala.sh
                rm -fv "$subsubdir"/nuts.py
                rm -fv "$subsubdir"/run-nuts.sh
            fi
        done
    fi
done

echo "Finishing job"
exit 0
