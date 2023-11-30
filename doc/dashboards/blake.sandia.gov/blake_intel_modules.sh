#!/bin/bash
module purge
module load git cmake
module load python/3.10.10
module load intel-oneapi-compilers/2023.2.0
module load metis/5.1.0
module load intel-oneapi-mkl/2023.2.0
module load superlu/5.3.0
#module load zlib/1.3 # Don't use. Error: "Relink ../libimf.so with /lib64/libm.so.6 for IFUNC symbol sincosf"
module load intel-oneapi-mpi/2021.10.0
module load hdf5/1.14.2
module load superlu-dist/8.1.2
module load parallel-netcdf/1.12.3
module load netcdf-c/4.9.2
module load netcdf-fortran/4.6.1
source /home/projects/albany/tpls/python/oneapi/2023.2.0/mpi/2021.10.0/oneapi-env/bin/activate # activate python env
ulimit -l unlimited # Unlimited memory locking. Error: map_hfi_mem: mmap of rcvhdr_bufbase size 262144 failed: Resource temporarily unavailable
module list
