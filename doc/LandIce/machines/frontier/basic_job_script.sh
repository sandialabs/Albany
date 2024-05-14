#!/bin/bash -login

#SBATCH -A <account>
#SBATCH --job-name=AISRun
#SBATCH --output=AISRun.%j.out
#SBATCH --error=AISRun.%j.err 
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --time=00:30:00 
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

# Load modules
SCRIPT_DIR=
source ${SCRIPT_DIR}/frontier_gpu_modules.sh

# Env variables
APPDIR=
export TPETRA_ASSUME_GPU_AWARE_MPI=1

# Log input settings to output
cat input_albany_MueLuKokkos.yaml

# Run case
srun ${APPDIR}/Albany input_albany_MueLuKokkos.yaml >& out-albany.txt
