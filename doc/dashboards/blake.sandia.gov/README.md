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
source blake_gcc_modules.sh
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
Edit the configuration script to point to the repo location and install directory:
```sh
REPO_DIR=${HOME}/Trilinos
INSTALL_DIR=${HOME}/Trilinos/trilinos-serial-gcc/install
```

Build and install Trilinos by using the configuration file provided:
```sh
source do-cmake-trilinos-gcc-serial-release
make -j
make install
```
