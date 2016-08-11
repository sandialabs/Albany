

#!/bin/bash

module unload cmake netcdf-hdf5parallel/4.2.0 python
module swap PrgEnv-intel PrgEnv-gnu; 
module load cmake/3.3.2 python cray-netcdf-hdf5parallel 
module load boost/1.61
module list
