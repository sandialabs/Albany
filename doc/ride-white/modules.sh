#!/bin/bash
module purge
module load cmake git
module load openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load boost/1.60.0/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load netcdf/4.4.1/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load netcdf-f/4.4.4/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44 
module load pnetcdf/1.6.1/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load hdf5/1.8.17/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module load openblas/0.2.19/gcc/5.4.0
module list
