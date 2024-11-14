# This file is for user convenience only and is not used by the model
# Changes to this file will be ignored and overwritten
# Changes to the environment should be made in env_mach_specific.xml
# Run ./case.setup --reset to regenerate this file
. /usr/share/lmod/8.3.1/init/sh
module unload cray-hdf5-parallel cray-netcdf-hdf5parallel cray-parallel-netcdf cray-netcdf cray-hdf5 PrgEnv-gnu PrgEnv-intel PrgEnv-nvidia PrgEnv-cray PrgEnv-aocc intel intel-oneapi nvidia aocc cudatoolkit climate-utils matlab craype-accel-nvidia80 craype-accel-host perftools-base perftools darshan
module load PrgEnv-gnu/8.3.3 gcc/11.2.0 cray-libsci/23.02.1.1 craype-accel-host craype/2.7.20 cray-mpich/8.1.25 cray-hdf5-parallel/1.12.2.3 cray-netcdf-hdf5parallel/4.9.0.3 cray-parallel-netcdf/1.12.3.3 cmake/3.24.3 evp-patch
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_MPIIO_DVS_MAXNODES=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export PERL5LIB=/global/cfs/cdirs/e3sm/perl/lib/perl5-only-switch
export FI_CXI_RX_MATCH_MODE=software
export MPICH_COLL_SYNC=MPI_Bcast
export NETCDF_PATH=/opt/cray/pe/netcdf-hdf5parallel/4.9.0.3/gnu/9.1
export PNETCDF_PATH=/opt/cray/pe/parallel-netcdf/1.12.3.3/gnu/9.1
export GATOR_INITIAL_MB=4000MB
export ADIOS2_ROOT=/global/cfs/cdirs/e3sm/3rdparty/adios2/2.9.1/cray-mpich-8.1.25/gcc-11.2.0
export BLA_VENDOR=Generic
export MOAB_ROOT=/global/cfs/cdirs/e3sm/software/moab/gnu
