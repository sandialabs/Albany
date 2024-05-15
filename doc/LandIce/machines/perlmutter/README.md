## Clone repos and load modules
These build instructions are for compiling Albany Land Ice (ALI) on Perlmutter at NERSC.

Download Trilinos and Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/trilinos/Trilinos.git
git clone https://github.com/sandialabs/Albany.git
```
for serial builds, load cpu modules:
```sh
source pm_cpu_gnu_modules.sh
```
for cuda builds, load gpu modules:
```sh
source pm_gpu_gnumodules.
```

## Building Trilinos/develop
Switch to the develop branch of Trilinos:
```sh
cd ${HOME}/Trilinos
git checkout develop
```

### For serial builds:
Make a build directory for Trilinos and copy over the configuration file provided:
```sh
mkdir trilinos-serial-gcc
cd trilinos-serial-gcc
cp ${HOME}/Albany/doc/LandIce/machines/perlmutter/do-cmake-trilinos-serial-gcc .
```
Edit the configuration script to point to headers for Boost (if on fanssie project, you can use the commented path), then edit the configuration file to point to the repository location and install directory:
```sh
BOOST_DIR=<boost directory>
TRILINOS_SOURCE_DIR=${HOME}/Trilinos
TRILINOS_INSTALL_DIR=${HOME}/Trilinos/trilinos-serial-gcc/install
```
Build and install Trilinos by using the configuration file provided:
```sh
source do-cmake-trilinos-serial-gcc
make -j
make install
```

### For gpu builds:
GPU builds have the option of enabling or disabling unified virtual memory and configuration files have been included for both cases. If you want to enable CUDA UVM, please replace `cuda` with `cudauvm` in the following instructions.

Make a build directory for Trilinos and copy over the configuration file provided:
```sh
mkdir trilinos-cuda-gcc
cd trilinos-cuda-gcc
cp ${HOME}/Albany/doc/LandIce/machines/perlmutter/do-cmake-trilinos-cuda-gcc .
```
Edit the configuration script to point to headers for Boost (if on fanssie project, you can use the commented path), then edit the configuration file to point to the repository location and install directory:
```sh
BOOST_DIR=<boost directory>
TRILINOS_SOURCE_DIR=${HOME}/Trilinos
TRILINOS_INSTALL_DIR=${HOME}/Trilinos/trilinos-serial-gcc/install
```

Compiling GPU builds also require editing the configuration script to point to the kokkos nvcc wrapper which is provided in this directory:
```sh
NVCC_WRAPPER=${HOME}/Albany/doc/LandIce/machines/perlmutter/nvcc_wrapper_a100
```

Build and install Trilinos by using the configuration file provided:
```sh
source do-cmake-trilinos-cuda-gcc
make -j
make install
```




