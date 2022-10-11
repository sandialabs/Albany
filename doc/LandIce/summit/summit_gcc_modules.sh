#!/bin/bash
module purge
module load hsi xalt lsf-tools darshan-runtime DefApps
module unload spectrum-mpi
module unload xl
module load gcc/9.1.0
module load spectrum-mpi/10.4.0.3-20210112
module load git cmake
#module load boost/1.76.0
module load boost/1.77.0
module load hdf5/1.10.7
#module load netcdf-c/4.8.0
module load netcdf-c/4.8.1
module load netcdf-fortran/4.4.5
module load parallel-netcdf/1.12.2
module load netlib-lapack/3.9.1
#module load cuda/10.1.243 # doesn't work with gcc 9.1.0
module load cuda/11.0.3
module list

