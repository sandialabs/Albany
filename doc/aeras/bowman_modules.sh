#!/bin/bash
# This script loads the necessary modules and sets the necessary 
# environment variables on the Bowman cluster for the compilation 
# of Trilinos and Albany/Aeras for Xeon Phi architectures.
#

module purge
module load cmake/3.5.2
module load intel/compilers/17.0.098
module load openmpi/1.10.4/intel/17.0.098
module load zlib/1.2.8
module load hdf5/1.8.17/openmpi/1.10.4/intel/17.0.098
module load pnetcdf/1.7.0/openmpi/1.10.4/intel/17.0.098
module load netcdf/4.4.1/openmpi/1.10.4/intel/17.0.098
module load boost/1.55.0/openmpi/1.10.4/intel/17.0.098
module list 
