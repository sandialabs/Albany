#!/bin/bash
module purge 
#module load devpack/20210310/openmpi/4.0.5/intel/oneapi/2021.1.1 # this one works / doesn't work anymore (some issue w/ kokkos kernels) / no boostlib
#module load devpack/20210420/openmpi/4.0.5/intel/oneapi/2021.2.0 # undefined reference to `ncmpi_strerror'
module load devpack/20190329/openmpi/4.0.1/intel/19.3.199 # linear solve is worse?
module swap cmake/3.22.2
#module swap cmake/3.12.3 cmake/3.19.3
#module load devpack/latest/openmpi/2.1.2/intel/18.1.163
#module swap cmake/3.9.0 cmake/3.19.3
module load python/3.7.3
#module load python/3.8.8/gcc/10.2.0
module list 
