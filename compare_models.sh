#!/bin/sh
#BSUB -J comp_models
#BSUB -R "rusage[mem=4GB]"
###BSUB -q gpua100
###BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 2
#BSUB -R "span[hosts=1]" 
#BSUB -o compare_models_%J.out
#BSUB -e compare_models_%J.err
#BSUB -W 10:00
# -- end of LSF options --

source diffwave/bin/activate
module load pandas/1.4.4-python-3.10.7
module load numpy/1.23.3-python-3.10.7-openblas-0.3.21
module load matplotlib/3.6.0-numpy-1.23.3-python-3.10.7
module load cuda


### python Christians_seje_hpc_scripts/test_gpu.py
python compare_models.py
