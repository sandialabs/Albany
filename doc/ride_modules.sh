
#Before these are loaded, one needs to get a job:
#salloc -t 8:00:00 -p rhel7F
module purge 
module load gcc/4.9.2
module load openmpi/1.10.2/gcc/4.9.2/cuda/7.5.7
module load boost/1.60.0/openmpi/1.10.2/gcc/4.9.2/cuda/7.5.7
module load hdf5/1.8.16/openmpi/1.10.2/gcc/4.9.2/cuda/7.5.7
module load netcdf-exo/4.3.3.1/openmpi/1.10.2/gcc/4.9.2/cuda/7.5.7
module load pnetcdf/1.6.1/openmpi/1.10.2/gcc/4.9.2/cuda/7.5.7
module load parmetis/4.0.3/openmpi/1.10.2/gcc/4.9.2/cuda/7.5.7
module load openblas/0.2.15/gcc/4.9.2
module list 
#FIXME: the following needs needs to be changed to point to your Trilinos! 
export OMPI_CXX=/home/ikalash/Trilinos/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1

