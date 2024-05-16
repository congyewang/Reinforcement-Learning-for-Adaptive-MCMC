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
                if [ -f "$subsubdir/run-mala.sh" ]; then
                    cd "$subsubdir"
                    echo "Update shell script in $subsubdir Dir"
                    sbatch run-baseline.sh
                    sbatch run-mala.sh
                    cd -
                else
                    echo "No run-mala.sh file in Dir $subsubdir"
                fi
            fi
        done
    fi
done

echo "Finishing job"
exit 0
