#!/bin/bash


BASELINES_DIR="./baselines"


if [ ! -d "$BASELINES_DIR" ]; then
    echo "Dir $BASELINES_DIR does not exist."
    exit 1
fi

for subdir in "$BASELINES_DIR"/*; do
    if [ -d "$subdir" ]; then
        if [ -f "$subdir/run-nuts.sh" ]; then
            cd "$subdir"
            echo "Update shell script in $subdir Dir"
            sbatch run-nuts.sh
            cd -
        else
            echo "No run-nuts.sh file in Dir $subdir"
        fi
    fi
done

echo "Finishing job"
echo 0
