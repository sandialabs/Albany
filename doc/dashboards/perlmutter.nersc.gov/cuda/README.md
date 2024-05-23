# Quick Start: Build Albany with weekly Trilinos
These build instructions are for compiling Albany Land Ice (ALI) on Perlmutter at NERSC using a CUDA backend with UVM-disabled. *Note: UVM-disabled is still in development so some features may result in illegal device access exceptions at runtime.*

Download Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/sandialabs/Albany.git
```
and load modules:
```sh
source ${HOME}/Albany/doc/dashboards/perlmutter.nersc.gov/cuda/pm_gpu_gnu_modules.sh
```
Make a build directory for Albany and copy over the configuration file provided:
```sh
cd ${HOME}/Albany
mkdir albany-cuda-gcc
cd albany-cuda-gcc
cp ${HOME}/Albany/doc/dashboards/perlmutter.nersc.gov/cuda/do-cmake-pm_gpu-albany .
```
Modify configuration file to point to Trilinos weekly install and where you want to install Albany as well as paths to MPI c compilers and NVCC compiler wrapper:
```sh
TRILINOS_INSTALL=/global/cfs/cdirs/fanssie/automated_testing/weeklyCDashPerlmutter/cuda/builds/TrilinosInstall
ALBANY_INSTALL=${HOME}/albany-cuda-gcc/install
NVCC_PATH=${HOME}/Albany/doc/dashboards/perlmutter.nersc.gov/cuda/nvcc_wrapper_a100
MPICC_PATH=${MPICH_DIR}/bin/mpicc
```
Configure and build/install Albany:
```sh
source do-cmake-pm_gpu-albany
make -j 10 install
```

# Build Trilinos/Albany
## Clone repos and load modules
Download Trilinos and Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/trilinos/Trilinos.git
git clone https://github.com/sandialabs/Albany.git
```
and load modules for hardware type of choice (`cpu` or `gpu`):
```sh
source ${HOME}/Albany/doc/dashboards/perlmutter.nersc.gov/cuda/pm_gpu_gnu_modules.sh
```

## Building Trilinos/develop
Switch to the develop branch of Trilinos, make a build directory for Trilinos and copy over the configuration file provided:
```sh
cd ${HOME}/Trilinos
git checkout develop
mkdir trilinos-cuda-gcc
cd trilinos-cuda-gcc
cp ${HOME}/Albany/doc/dashboards/perlmutter.sandia.gov/cuda/do-cmake-pm_gpu-trilinos .
```
Update configuration file to set where you want to install Trilinos:
```sh
INSTALL_DIR=${HOME}/trilinos-cuda-gcc/install
```
Build and install Trilinos on a compute node:
```sh
source do-cmake-pm_gpu-trilinos
make -j 10 install
```

## Building Albany/master
Make a build directory for Albany and copy over the configuration file provided:
```sh
cd ${HOME}/Albany
mkdir albany-cuda-gcc
cd albany-cuda-gcc
cp ${HOME}/Albany/doc/dashboards/perlmutter.nersc.gov/cuda/do-cmake-pm_gpu-albany .
```
Modify configuration file to point to Trilinos install location and where you want to install Albany as well as paths to MPI c compilers and NVCC compiler wrapper:
```sh
TRILINOS_INSTALL=${HOME}/trilinos-cuda-gcc/install
ALBANY_INSTALL=${HOME}/albany-cuda-gcc/install
NVCC_PATH=${HOME}/Albany/doc/dashboards/perlmutter.nersc.gov/cuda/nvcc_wrapper_a100
MPICC_PATH=${MPICH_DIR}/bin/mpicc
```
Configure and build/install Albany:
```sh
source do-cmake-pm_gpu-albany
make -j 10 install
```