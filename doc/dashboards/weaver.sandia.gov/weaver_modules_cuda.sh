
module purge 
module load git
#module load devpack/20180517/openmpi/2.1.2/gcc/7.2.0/cuda/9.2.88
module load devpack/20190814/openmpi/4.0.1/gcc/7.2.0/cuda/10.1.105
module load yamlcpp/0.5.3/gcc/7.2.0
module unload cmake 
module load cmake/3.19.3 
module list
#FIXME: the following needs needs to be changed to point to your Trilinos!
#export OMPI_CXX=${jenkins_trilinos_dir}/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1
 
