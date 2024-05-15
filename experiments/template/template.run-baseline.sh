#!/bin/bash
#
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=5

module load GCC/6.3.0-2.27

module load MATLAB
matlab -nodisplay -nosplash < baseline.m

echo Finishing job
exit 0
