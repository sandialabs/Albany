
module load git 
module add netcdf/4.4.1/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44 netcdf-f/4.4.4/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44 openblas/0.2.19/gcc/5.4.0 zlib/1.2.8
module add cuda/8.0.44 openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44 cmake
module load boost/1.60.0/openmpi/1.10.4/gcc/5.4.0/cuda/8.0.44
module list
#FIXME: the following needs needs to be changed to point to your Trilinos!
export OMPI_CXX=${jenkins_trilinos_dir}/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1
 
