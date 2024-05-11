#!/bin/bash
#SBATCH --mail-type=FAIL

module load GCC/6.3.0-2.27

source /nobackup/c2029946/Software/miniconda3/bin/activate rl
python mala.py

echo Finishing job
exit 0
