# This file is for user convenience only and is not used by the model
# Changes to this file will be ignored and overwritten
# Changes to the environment should be made in env_mach_specific.xml
# Run ./case.setup --reset to regenerate this file
. /gpfs/fs1/soft/chrysalis/spack/opt/spack/linux-centos8-x86_64/gcc-9.3.0/lmod-8.3-5be73rg/lmod/lmod/init/sh
module purge 
module load subversion/1.14.0-e4smcy3 perl/5.32.0-bsnc6lt cmake/3.24.2-whgdv7y intel/20.0.4-kodw73g intel-mkl/2020.4.304-g2qaxzf openmpi/4.1.6-2mm63n2 hdf5/1.10.7-4cghwvq netcdf-c/4.4.1-a4hji6e netcdf-cxx/4.2-ldoxr43 netcdf-fortran/4.4.4-husened parallel-netcdf/1.11.0-icrpxty
export PERL5LIB=/lcrc/group/e3sm/soft/perl/chrys/lib/perl5
export NETCDF_C_PATH=/gpfs/fs1/soft/chrysalis/spack/opt/spack/linux-centos8-x86_64/intel-20.0.4/netcdf-c-4.4.1-a4hji6e
export NETCDF_FORTRAN_PATH=/gpfs/fs1/soft/chrysalis/spack/opt/spack/linux-centos8-x86_64/intel-20.0.4/netcdf-fortran-4.4.4-husened
export PNETCDF_PATH=/gpfs/fs1/soft/chrysalis/spack/opt/spack/linux-centos8-x86_64/intel-20.0.4/parallel-netcdf-1.11.0-icrpxty
export OMPI_MCA_sharedfp=^lockedfile,individual
export UCX_TLS=^xpmem
export OMP_STACKSIZE=128M
export KMP_AFFINITY=granularity=core,balanced
export MOAB_ROOT=/lcrc/soft/climate/moab/chrysalis/intel
