
module purge
module load cmake/2.8.11.2
module load openmpi/1.10.1/gnu/4.7.2/cuda/7.5.7
module load intel/15.5.223
module load nvcc-wrapper/gnu
module list 
export CC=mpicc
export CXX=mpicxx
export FC=mpif90
export NVCC_WRAPPER_DEFAULT_COMPILER=mpicc
#FIXME: the following needs needs to be changed to point to your Trilinos! 
export OMPI_CXX=/home/jwatkin/Trilinos/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1

