#!/bin/bash
module load PrgEnv-gnu
module load cpe-cuda # get correct cuda-gcc compiler
module load cmake
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load cray-parallel-netcdf
module swap cuda/11.3.0 cuda/11.1.1
module list

