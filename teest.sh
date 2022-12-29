#!/bin/sh
#BSUB -J experiment
#BSUB -R "rusage[mem=4GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]" 
#BSUB -o teest.out
#BSUB -e teest.err
#BSUB -W 24:00
# -- end of LSF options --

source diffwave/bin/activate
module load pandas/1.4.4-python-3.10.7
module load numpy/1.23.3-python-3.10.7-openblas-0.3.21
module load matplotlib/3.6.0-numpy-1.23.3-python-3.10.7
module load cuda


### python Christians_seje_hpc_scripts/test_gpu.py
python teeeeeest.py
