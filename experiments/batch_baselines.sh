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
                if [ -f "$subsubdir/run-nuts.sh" ]; then
                    cd "$subsubdir"
                    echo "Update shell script in $subsubdir Dir"
                    sbatch run-mala.sh
                    sbatch run-nuts.sh
                    cd -
                else
                    echo "No run-nuts.sh file in Dir $subsubdir"
                fi
            fi
        done
    fi
done

echo "Finishing job"
echo 0
