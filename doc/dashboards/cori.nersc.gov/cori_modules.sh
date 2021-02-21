

#!/bin/bash


module unload cmake cray-netcdf-hdf5parallel python 
module unload intel PrgEnv-intel 
module load PrgEnv-gnu
module unload cray-mpich
module load cray-mpich/7.7.6 
module unload gcc/7.3.0 
module load gcc/8.3.0 
module load boost 
module load cmake/3.18.2 cray-netcdf-hdf5parallel
module load cray-python/2.7.15.6
module list
