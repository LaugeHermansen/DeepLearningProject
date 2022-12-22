#BSUB -J test_hpc_J

#BSUB -R "rusage[mem=1024MB]"

python test_hpc.py
