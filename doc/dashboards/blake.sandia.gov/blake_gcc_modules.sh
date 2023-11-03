#!/bin/bash
module purge
module load git cmake
module load python/3.10.10
export PATH=${HOME}/.local/bin:${PATH} # Append local python install bin to path for pip3
module load gcc/12.2.0
module load boost/1.82.0
module load metis/5.1.0
module load netlib-lapack/3.11.0
module load superlu/5.3.0
module load zlib/1.2.13
module load openmpi/4.1.5
module load hdf5/1.14.1-2
module load superlu-dist/8.1.2
module load parallel-netcdf/1.12.3
module load netcdf-c/4.9.2
module load netcdf-fortran/4.6.0
module list
