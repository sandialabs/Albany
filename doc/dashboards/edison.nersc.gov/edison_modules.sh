

#!/bin/bash

module unload cmake netcdf-hdf5parallel/4.2.0 python
module swap PrgEnv-intel PrgEnv-gnu; 
module load cmake 
module load python 
module load cray-netcdf-hdf5parallel 
#module load usg-default-modules/1.1
module load boost
module list
