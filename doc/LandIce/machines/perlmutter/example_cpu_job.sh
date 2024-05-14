#!/bin/bash -login

#SBATCH -A m4274
#SBATCH --job-name=Albany
#SBATCH --output=Albany.cpu.%j.out
#SBATCH --error=Albany.cpu.%j.err
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-socket=64
#SBATCH --cpu-bind=cores
#SBATCH --hint=nomultithread

# Env variables
SCRIPT_DIR=
ALBANY_INSTALL=
TRILINOS_INSTALL=
APPDIR=${ALBANY_INSTALL}/bin

# Load modules
source ${SCRIPT_DIR}/pm_cpu_gnu_modules.sh

# Add libraries to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ALBANY_INSTALL}/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TRILINOS_INSTALL}/lib64

# Run Case
srun ${APPDIR}/Albany input_albany_MueLu.yaml
