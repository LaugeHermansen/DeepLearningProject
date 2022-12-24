#BSUB -J test_hpc_J
#BSUB -R "rusage[mem=2G]"

python main.py
