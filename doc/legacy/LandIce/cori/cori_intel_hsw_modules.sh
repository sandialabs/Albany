#!/bin/bash
# Notes: Once compiled, move executable to $SCRATCH for optimal performance
module rm cray-libsci # If using Intel MKL
module load git cmake
module load boost/1.70.0
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load cray-parallel-netcdf
module load metis/5.1.0
module unload craype-hugepages2M # caused some linking issues w/ mpas and albany
module list
