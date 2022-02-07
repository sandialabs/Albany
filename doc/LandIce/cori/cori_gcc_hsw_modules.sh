#!/bin/bash
# Notes: Once compiled, move executable to $SCRATCH for optimal performance
module swap PrgEnv-intel PrgEnv-gnu 
module load git cmake/3.18.2
module load boost/1.70.0
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load cray-parallel-netcdf
#module unload craype-hugepages2M # caused some linking issues w/ pio
module list
