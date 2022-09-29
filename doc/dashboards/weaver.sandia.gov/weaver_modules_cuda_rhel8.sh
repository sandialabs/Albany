
module purge 
module load git
#module load devpack/20210226/openmpi/4.0.5/gcc/7.2.0/cuda/10.2.2 
module load devpack/20190814/openmpi/4.0.1/gcc/7.2.0/cuda/10.1.105
module load yamlcpp/0.5.3/gcc/7.2.0
module load python/3.7.3
module list
#FIXME: the following needs needs to be changed to point to your Trilinos!
#export OMPI_CXX=${jenkins_trilinos_dir}/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1
 
