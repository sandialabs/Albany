#!/bin/bash -login

#SBATCH -A <account>
#SBATCH --job-name=Albany
#SBATCH --output=Albany.%j.out
#SBATCH --error=Alabny.%j.err
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=00:20:00 
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH --gpu-bind=none

# Env variables
SCRIPT_DIR=
ALBANY_INSTALL=
TRILINOS_INSTALL=
APPDIR=${ALBANY_INSTALL}/bin

# Load modules
source ${SCRIPT_DIR}/pm_gpu_gnu_modules.sh

# Add libraries to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ALBANY_INSTALL}/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TRILINOS_INSTALL}/lib64

# Run case
srun bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); ${APPDIR}/Albany --kokkos-map-device-id-by=mpi_rank input.yaml"
