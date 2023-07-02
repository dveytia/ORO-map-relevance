#!/bin/bash

#SBATCH --job-name=predictions-classifier-excl

#SBATCH --output=predictions-classifier-excl.%J.out
#SBATCH --error=predictions-classifier-excl.%J.err
#SBATCH --ntasks=10
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --account=scw1598
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vcm20gly@bangor.ac.uk  

module purge
module load mpi
module load anaconda/3
source activate review

mpiexec -n 10 python binary_predictions_excl.py
