

#!/bin/bash

module unload cmake netcdf-hdf5parallel/4.2.0 python
module swap PrgEnv-pgi PrgEnv-gnu; 
module load cmake/3.1.3 python cray-netcdf-hdf5parallel usg-default-modules/1.2
module load boost/1.57
module list
