#!/bin/bash
module unload cmake cray-netcdf-hdf5parallel python 
module unload intel PrgEnv-intel 
module load PrgEnv-gnu
module unload cray-mpich
module load cray-mpich/7.7.19
module unload gcc
module load gcc/8.3.0
module load cray-netcdf-hdf5parallel
module load cray-python/2.7.15.7
module list
