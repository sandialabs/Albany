

#!/bin/bash


module unload cmake cray-netcdf-hdf5parallel python 
module unload intel PrgEnv-intel 
module load PrgEnv-gnu
module unload cray-mpich
module load cray-mpich/7.7.6 
module unload gcc/7.3.0 
module load gcc/8.2.0 
module load boost/1.67.0 
module load git 
module load cmake/3.11.4 python cray-netcdf-hdf5parallel
module list
