
module purge
module load git  
module load cmake/3.0.2
module load gcc/4.9.2
module load openmpi/1.10.0/gnu/4.9.2/cuda/7.5.7
module load nvcc_wrapper/gnu
module load cuda/7.5.7
module load blas/3.5.0/gnu/4.9.2
module load lapack/3.5.0/gnu/4.9.2
module load netcdf/4.3.3.1/openmpi/1.10.0/gnu/4.9.2/cuda/7.5.7
module load boost/1.59.0/openmpi/1.10.0/gnu/4.9.2/cuda/7.5.7
module load hdf5/1.8.16/openmpi/1.10.0/gnu/4.9.2/cuda/7.5.7
module load pnetcdf/1.6.1/openmpi/1.10.0/gnu/4.9.2/cuda/7.5.7
module list 
export OMPI_CXX=/home/ikalash/Trilinos/packages/kokkos/config/nvcc_wrapper
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export CUDA_LAUNCH_BLOCKING=1

