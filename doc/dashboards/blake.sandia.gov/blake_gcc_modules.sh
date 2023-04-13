#!/bin/bash
module purge
module load git/2.9.4
module load cmake/3.25.2
module load python/3.7.3
module load gcc/11.2.0
module load boost/1.81.0/gcc/11.2.0
module load metis/5.1.0/gcc/11.2.0
module load netlib-lapack/3.10.1/gcc/11.2.0
module load superlu/5.3.0/gcc/11.2.0
module load zlib/1.2.13/gcc/11.2.0
module load openmpi/3.1.6/gcc/11.2.0
module load hdf5/1.14.0/gcc/11.2.0/openmpi/3.1.6
module load superlu-dist/8.1.2/gcc/11.2.0/openmpi/3.1.6
module load parallel-netcdf/1.12.3/gcc/11.2.0/openmpi/3.1.6
module load netcdf-c/4.9.0/gcc/11.2.0/openmpi/3.1.6
module list
