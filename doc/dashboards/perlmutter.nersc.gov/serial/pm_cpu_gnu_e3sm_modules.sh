#!/bin/bash
# --- Unload Modules ---
MODULES_TO_UNLOAD=(
    "cpe"
    "cray-hdf5-parallel"
    "cray-netcdf-hdf5parallel"
    "cray-parallel-netcdf"
    "cray-netcdf"
    "cray-hdf5"
    "PrgEnv-gnu"
    "PrgEnv-intel"
    "PrgEnv-nvidia"
    "PrgEnv-cray"
    "PrgEnv-aocc"
    "gcc-native"
    "intel"
    "intel-oneapi"
    "nvidia"
    "aocc"
    "cudatoolkit"
    "climate-utils"
    "cray-libsci"
    "matlab"
    "craype-accel-nvidia80"
    "craype-accel-host"
    "perftools-base"
    "perftools"
    "darshan"
)

for mod in "${MODULES_TO_UNLOAD[@]}"; do
    module unload "${mod}" 2>/dev/null || true
done

# --- Load Modules ---
module load PrgEnv-gnu/8.6.0
module load gcc-native/14
module load cray-libsci/25.09.0
module load craype-accel-host
module load craype/2.7.35
module load cray-mpich/9.0.1
module load cray-hdf5-parallel/1.14.3.7
module load cray-netcdf-hdf5parallel/4.9.2.1
module load cray-parallel-netcdf/1.12.3.19
module load cmake/3.30.2

# --- Environment Variables ---
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_MPIIO_DVS_MAXNODES=1
export HDF5_USE_FILE_LOCKING=FALSE
export FI_MR_CACHE_MONITOR=kdreg2
export NETCDF_PATH=${CRAY_NETCDF_HDF5PARALLEL_PREFIX}
export PNETCDF_PATH=${CRAY_PARALLEL_NETCDF_PREFIX}
export GATOR_INITIAL_MB=4000MB
export MPICH_SMP_SINGLE_COPY_MODE=CMA
export STK_UNKNOWN_PATTERN_EXCHANGER=Prepost
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# --- System Resources ---
# RLIMIT_STACK = -1 typically maps to unlimited
ulimit -s unlimited

# Extras
export SUPERLU_DIR=/global/common/software/fanssie/superlu/7.0.1/gcc/14.3
export BOOST_DIR=/global/common/software/fanssie/boost/1.83.0/gcc/14.3
unset MPICH_GPU_SUPPORT_ENABLED

