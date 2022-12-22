#!/bin/sh
#BSUB -q gpuv100
#BSUB -J experiment_from_bottom
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]" 
#BSUB -o experiment_from_bottom_%J.out
#BSUB -e experiment_from_bottom_%J.err
#BSUB -W 24:00
# -- end of LSF options --

export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC

source ~/.bashrc
conda activate diffwave

python experiment_from_bottom.py --exp_name "from_bottom" --seed 58008 --ckpt "849"
