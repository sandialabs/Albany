# Quick Start: Build Albany with nightly Trilinos
These build instructions are for compiling Albany Land Ice (ALI) on weaver at Sandia National Laboratories.

Download Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/sandialabs/Albany.git
```
and load modules:
```sh
source ${HOME}/Albany/doc/dashboards/weaver.sandia.gov/weaver_modules_cuda.sh
```
Make a build directory for Albany and copy over the configuration file provided:
```sh
cd ${HOME}/Albany
mkdir albany-cuda-gcc
cd albany-cuda-gcc
cp ${HOME}/Albany/doc/dashboards/weaver.sandia.gov/do-cmake-albany .
```
Configure and build Albany on a compute node:
```sh
bsub -Is -gpu num=4 -n 40 bash
source do-cmake-albany
make -j 40
exit
```

# Build Trilinos/Albany
## Clone repos and load modules
Download Trilinos and Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/trilinos/Trilinos.git
git clone https://github.com/sandialabs/Albany.git
```
and load modules:
```sh
source ${HOME}/Albany/doc/dashboards/weaver.sandia.gov/weaver_modules_cuda.sh
```

## Building Trilinos/develop
Switch to the develop branch of Trilinos, make a build directory for Trilinos and copy over the configuration files provided:
```sh
cd ${HOME}/Trilinos
git checkout develop
mkdir trilinos-cuda-gcc
cd trilinos-cuda-gcc
cp ${HOME}/Albany/doc/dashboards/weaver.sandia.gov/do-cmake-weaver-trilinos .
cp ${HOME}/Albany/doc/dashboards/weaver.sandia.gov/nvcc_wrapper_volta .
```
Build and install Trilinos on a compute node:
```sh
bsub -Is -gpu num=4 -n 40 bash
source do-cmake-weaver-trilinos
make -j 40
make install
exit
```

## Building Albany/master
Make a build directory for Albany and copy over the configuration file provided:
```sh
cd ${HOME}/Albany
mkdir albany-cuda-gcc
cd albany-cuda-gcc
cp ${HOME}/Albany/doc/dashboards/weaver.sandia.gov/do-cmake-albany .
```
Edit the configuration script to point to the trilinos install directory, albany install directory and nvcc_wrapper location:
```sh
TRILINOS_INSTALL=${HOME}/Trilinos/albany-cuda-gcc/install
ALBANY_INSTALL=${HOME}/Albany/albany-cuda-gcc/install
NVCC_WRAPPER=${HOME}/Trilinos/albany-cuda-gcc/nvcc_wrapper_volta
```
Configure, build and install Albany on a compute node:
```sh
bsub -Is -gpu num=4 -n 40 bash
source do-cmake-albany
make -j 40
make install
exit
```
