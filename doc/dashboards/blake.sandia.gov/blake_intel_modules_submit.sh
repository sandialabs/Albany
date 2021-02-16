#!/bin/bash
module purge 
module load devpack/latest/openmpi/2.1.2/intel/18.1.163
module unload cmake
module load cmake/3.19.3
module list 
