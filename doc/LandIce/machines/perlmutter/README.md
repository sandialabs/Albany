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
source pm_gpu_gnu_modules.sh
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
Edit the configuration script to point to headers for Boost (an installation is available for members of fanssie in the commented path), then edit the configuration file to point to the repository location and install directory:
```sh
BOOST_DIR=<boost directory>
TRILINOS_SOURCE_DIR=${HOME}/Trilinos
TRILINOS_INSTALL_DIR=${HOME}/trilinos-serial-gcc/install
```
Build and install Trilinos by using the configuration file provided:
```sh
source do-cmake-trilinos-serial-gcc
make -j 10 install
```

### For gpu builds:
GPU builds have the option of enabling or disabling unified virtual memory and configuration files have been included for both cases. If you want to disable CUDA UVM, please replace `cudauvm` with `cuda` in the following instructions.

Make a build directory for Trilinos and copy over the configuration file provided:
```sh
mkdir trilinos-cudauvm-gcc
cd trilinos-cudauvm-gcc
cp ${HOME}/Albany/doc/LandIce/machines/perlmutter/do-cmake-trilinos-cudauvm-gcc .
```
Edit the configuration script to point to headers for Boost (an installation is available for members of fanssie in the commented path), then edit the configuration file to point to the repository location and install directory:
```sh
BOOST_DIR=<boost directory>
TRILINOS_SOURCE_DIR=${HOME}/Trilinos
TRILINOS_INSTALL_DIR=${HOME}/trilinos-cudauvm-gcc/install
```

Compiling GPU builds also require editing the configuration script to point to the kokkos nvcc wrapper which is provided in this directory:
```sh
NVCC_WRAPPER=${HOME}/Albany/doc/LandIce/machines/perlmutter/nvcc_wrapper_a100
```

Build and install Trilinos by using the configuration file provided:
```sh
source do-cmake-trilinos-cudauvm-gcc
make -j 10 install
```
## Building Albany

*A note on Fad sizes*: Albany uses Sacado Fad objects for doing automatic differentiation. When configuring a build of Albany, the size and type for these objects should be supplied. Configuration scripts have been included here for two cases: 
1) `*-sfad` corresponds to SFad types with a fixed size that can be supplied (the provided scripts set this to 12 but will be problem dependent. Sfad size depends on number of equations and type of elements, for wedge elements the size should be `6*number_of_equations` and for hex elements it should be `8*number_of_equations`), 
2) `*-slfad` corresponds to SLFad types which will have some maximum number of components at compile-time but the actual number used will be chosen at run-time. This option will work for any case but will suffer a performance penalty.
3) `*-dfad` corresponds to DFad types which are completely set at run-time and will also have a performance penalty. This option will only work for serial builds of Albany.
This readme will assume that we are using the default sfad scripts.

### For GPU builds:
Make a build directory for Albany and copy over the configuration file provided:
```sh
mkdir albany-cuda-gcc-sfad12
cd albany-cuda-gcc-sfad12
cp ${HOME}/Albany/doc/LandIce/machines/perlmutter/do-cmake-albany-cuda-gcc-sfad .
```

Edit the configuration script to:
```sh
TRILINOS_INSTALL_DIR=${HOME}/Trilinos/trilinos-cudauvm-gcc/install
ALBANY_SOURCE_DIR=${HOME}/Albany
NVCC_WRAPPER=${HOME}/Albany/doc/LandIce/machines/perlmutter/nvcc_wrapper_a100
ALBANY_INSTALL_DIR=${HOME}/albany-cuda-gcc-sfad12/install
```
Build and install Albany by using the configuration file provided:
```sh
source do-cmake-albany-cuda-gcc-sfad
make -j 10 install
```
