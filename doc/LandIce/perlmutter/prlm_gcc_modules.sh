#!/bin/bash
module load PrgEnv-gnu
#module load cudatoolkit/21.3_11.2
module load cudatoolkit/21.9_11.0
#module load cudatoolkit/21.9_11.4 # STK T0 error
module load cpe-cuda # get correct cuda-gcc compiler
module load cmake
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load cray-parallel-netcdf
#module swap cuda/11.3.0 cuda/11.1.1
module load craype-accel-nvidia80
module list

