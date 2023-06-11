#!/bin/bash

#SBATCH --job-name=model-selection-excl

#SBATCH --output=model-selection-excl.%J.out
#SBATCH --error=model-selection-excl.%J.err
#SBATCH --nodes=2
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --account=scw1598
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vcm20gly@bangor.ac.uk  

module purge
module load mpi
module load anaconda/3
source activate review

mpiexec -n 5 python model_selection_excl.py 
