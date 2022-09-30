#!/bin/bash
module load PrgEnv-gnu
module load cudatoolkit
module load cpe-cuda # get correct cuda-gcc compiler
module load cmake
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load cray-parallel-netcdf
module load craype-accel-nvidia80
module load boost
module list
