

#!/bin/bash

module unload cmake netcdf-hdf5parallel/4.2.0 python usg-default-modules/1.1
module swap PrgEnv-pgi PrgEnv-gnu; module load cmake/2.8.12.1 python netcdf-hdf5parallel/4.3.0 usg-default-modules/1.0
module load boost
module list
