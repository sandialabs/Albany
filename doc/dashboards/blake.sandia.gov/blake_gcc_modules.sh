#!/bin/bash
module purge
module load git cmake
module load python/3.10.10
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
module load parmetis/4.0.3
source /home/projects/albany/tpls/python/gcc/12.2.0/openmpi/4.1.5/gcc-env/bin/activate # activate python env
module list
