#!/bin/bash
#SBATCH --mail-type=FAIL

module load GCC/6.3.0-2.27

python mala.py

echo Finishing job
exit 0
