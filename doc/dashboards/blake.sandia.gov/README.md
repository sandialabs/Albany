# Quick Start: Build Albany with nightly Trilinos
These build instructions are for compiling Albany Land Ice (ALI) on blake at Sandia National Laboratories.

Download Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/sandialabs/Albany.git
```
and load modules:
```sh
source ${HOME}/Albany/doc/dashboards/blake.sandia.gov/blake_gcc_modules.sh
```
Make a build directory for Albany and copy over the configuration file provided:
```sh
cd ${HOME}/Albany
mkdir albany-serial-gcc
cd albany-serial-gcc
cp ${HOME}/Albany/doc/dashboards/blake.sandia.gov/do-cmake-albany-gcc-release .
```
Configure and build Albany on a compute node:
```sh
source do-cmake-albany-gcc-release
salloc -N1
make -j 96
exit
```

# Build Trilinos/Albany
## Clone repos and load modules
These build instructions are for compiling Albany Land Ice (ALI) on blake at Sandia National Laboratories.

Download Trilinos and Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/trilinos/Trilinos.git
git clone https://github.com/sandialabs/Albany.git
```
and load modules for compiler of choice (this document will assume gcc):
```sh
source ${HOME}/Albany/doc/dashboards/blake.sandia.gov/blake_gcc_modules.sh
```

## Building Trilinos/develop
Switch to the develop branch of Trilinos, make a build directory for Trilinos and copy over the configuration file provided:
```sh
cd ${HOME}/Trilinos
git checkout develop
mkdir trilinos-serial-gcc
cd trilinos-serial-gcc
cp ${HOME}/Albany/doc/dashboards/blake.sandia.gov/do-cmake-trilinos-gcc-release .
```
Build and install Trilinos on a compute node:
```sh
source do-cmake-trilinos-gcc-serial-release
salloc -N1
make -j 96
make install
exit
```

## Building Albany/master
Make a build directory for Albany and copy over the configuration file provided:
```sh
cd ${HOME}/Albany
mkdir albany-serial-gcc
cd albany-serial-gcc
cp ${HOME}/Albany/doc/dashboards/blake.sandia.gov/do-cmake-albany-gcc-release .
```
Edit the configuration script to point to the trilinos install directory and albany install directory:
```sh
TRILINOS_INSTALL=${HOME}/Trilinos/albany-serial-gcc/install
ALBANY_INSTALL=${HOME}/Albany/albany-serial-gcc/install
```
Configure, build and install Albany on a compute node:
```sh
source do-cmake-albany-gcc-release
salloc -N1
make -j 96
make install
exit
```
