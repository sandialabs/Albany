

#!/bin/bash

module unload gcc/7.3.0
module load boost/1.67.0
module load git 
module unload cmake netcdf-hdf5parallel/4.2.0 python
#module swap PrgEnv-intel PrgEnv-gnu; 
module unload intel PrgEnv-intel 
module load PrgEnv-gnu
module load cmake/3.11.4 python cray-netcdf-hdf5parallel
module load gcc/7.3.0 
#module unload darshan
module list
