# This file is for user convenience only and is not used by the model
# Changes to this file will be ignored and overwritten
# Changes to the environment should be made in env_mach_specific.xml
# Run ./case.setup --reset to regenerate this file
. /gpfs/fs1/soft/chrysalis/spack/opt/spack/linux-centos8-x86_64/gcc-9.3.0/lmod-8.3-5be73rg/lmod/lmod/init/sh
module purge 
module load subversion/1.14.0-e4smcy3 perl/5.32.0-bsnc6lt cmake/3.24.2-whgdv7y gcc/9.2.0-ugetvbp intel-mkl/2020.4.304-n3b5fye openmpi/4.1.3-sxfyy4k hdf5/1.10.7-j3zxncu netcdf-c/4.4.1-7ohuiwq netcdf-cxx/4.2-tkg465k netcdf-fortran/4.4.4-k2zu3y5 parallel-netcdf/1.11.0-mirrcz7
export PERL5LIB=/lcrc/group/e3sm/soft/perl/chrys/lib/perl5
export NETCDF_C_PATH=/gpfs/fs1/soft/chrysalis/spack/opt/spack/linux-centos8-x86_64/gcc-9.2.0/netcdf-c-4.4.1-7ohuiwq
export NETCDF_FORTRAN_PATH=/gpfs/fs1/soft/chrysalis/spack/opt/spack/linux-centos8-x86_64/gcc-9.2.0/netcdf-fortran-4.4.4-k2zu3y5
export PNETCDF_PATH=/gpfs/fs1/soft/chrysalis/spack/opt/spack/linux-centos8-x86_64/gcc-9.2.0/parallel-netcdf-1.11.0-mirrcz7
export OMPI_MCA_sharedfp=^lockedfile,individual
export UCX_TLS=^xpmem
export MOAB_ROOT=/lcrc/soft/climate/moab/chrysalis/gnu
