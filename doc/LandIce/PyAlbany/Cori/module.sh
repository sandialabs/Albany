module purge
source /opt/modules/default/init/sh
module rm craype craype-mic-knl craype-haswell PrgEnv-intel PrgEnv-cray PrgEnv-gnu intel cce gcc cray-parallel-netcdf cray-hdf5-parallel pmi cray-mpich2 cray-mpich cray-netcdf cray-hdf5 cray-netcdf-hdf5parallel cray-libsci papi cmake cray-petsc esmf zlib craype-hugepages2M darshan
module load craype PrgEnv-intel cray-mpich
module swap cray-mpich cray-mpich/7.7.10
module load PrgEnv-intel/6.0.5
module rm intel
module load intel/18.0.1.163
module swap craype craype/2.5.18
module rm pmi
module load pmi/5.0.14
module rm cray-netcdf-hdf5parallel
module load cray-netcdf-hdf5parallel/4.6.3.2 cray-hdf5-parallel/1.10.5.2 cray-parallel-netcdf/1.11.1.1
module rm git
module load git
module rm cmake
module load cmake/3.18.2
module unloadn cray-libsci
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export OMP_STACKSIZE=128M
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export MPICH_GNI_DYNAMIC_CONN=disabled
export FORT_BUFFERED=yes
export MPICH_MEMORY_REPORT=1
export CRAYPE_LINK_TYPE=dynamic
export CRAY_CPU_TARGET=haswell

conda activate my_pytrilinos_env
export PYTHONPATH="/global/homes/k/knliege/shared/trilinos/lib/python3.8/site-packages":$PYTHONPATH
export PYTHONPATH="/global/homes/k/knliege/shared/albany/lib/python3.8/site-packages":$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/homes/k/knliege/shared/trilinos/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/homes/k/knliege/shared/albany/lib/