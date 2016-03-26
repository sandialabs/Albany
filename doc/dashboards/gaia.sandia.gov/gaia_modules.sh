#!/bin/bash

module unload python/3.4.2
module unload gcc/5.1.0/openmpi/1.6.5
module unload gcc/5.1.0/base

module load cmake/2.8.12
module load gcc/5.1.0/openmpi/1.6.5
module load python/2.7.9
module load hdf5/1.8.12/gcc/5.1.0/openmpi/1.6.5
module load netcdf/4.3.2/gcc/5.1.0/parallel
module load boost/1.58.0/gcc/5.1.0

module list
