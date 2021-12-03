#!/bin/bash
module purge
module load hsi xalt lsf-tools darshan-runtime DefApps
module unload spectrum-mpi
module unload xl
module load gcc/6.4.0
module load spectrum-mpi/10.3.0.1-20190611
module load git/2.20.1 cmake/3.15.2
module load boost/1.66.0
module load hdf5/1.10.3
module load netcdf/4.6.1
module load parallel-netcdf/1.8.1
module load netlib-lapack/3.8.0
module load cuda/10.1.168
module list

