#!/bin/sh
#BSUB -q gpuv100
#BSUB -J test_models
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]" 
#BSUB -o test_models_%J.out
#BSUB -e test_models_%J.err
#BSUB -W 4:00
# -- end of LSF options --

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

source ~/.bashrc
conda activate diffwave

python test_models.py --test_mode 1

