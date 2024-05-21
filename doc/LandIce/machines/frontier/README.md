## Clone repos and load modules
These build instructions are for compiling Albany Land Ice (ALI) on Frontier at OLCF.

Download Trilinos and Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/trilinos/Trilinos.git
git clone https://github.com/sandialabs/Albany.git
```
Load gpu modules:
```sh
source frontier_gpu_modules.sh
```

## Building Trilinos/develop
Switch to the develop branch of Trilinos:
```sh
cd ${HOME}/Trilinos
git checkout develop
```

### Building Trilinos/develop:
Make a build directory for Trilinos and copy over the configuration file provided:
```sh
mkdir trilinos-rocm-gcc
cd trilinos-rocm-gcc
cp ${HOME}/Albany/doc/LandIce/machines/frontier/do-cmake-trilinos-gcc-hip .
```
Edit the configuration script to point to Trilinos source code and where you want to install Trilinos:
```sh
TRILINOS_SOURCE_DIR=${HOME}/Trilinos
TRILINOS_INSTALL_DIR=${HOME}/trilinos-rocm-gcc/install
```

Build and install Trilinos by using the configuration file provided:
```sh
source do-cmake-trilinos-gcc-hip
make -j 10 install
```
## Building Albany
Make a build directory for Albany and copy over the configuration file provided:
```sh
mkdir albany-rocm-gcc
cd albany-rocm-gcc
cp ${HOME}/Albany/doc/LandIce/machines/frontier/do-cmake-albany-gcc-hip .
```

Edit the configuration script to point to Trilinos install, the Albany source code, and where you want to install Albany:
```sh
TRILINOS_INSTALL_DIR=${HOME}/Trilinos/trilinos-rocm-gcc/install
ALBANY_SOURCE_DIR=${HOME}/Albany
ALBANY_INSTALL_DIR=${HOME}/albany-rocm-gcc/install
```
Build and install Albany by using the configuration file provided:
```sh
source do-cmake-albany-gcc-hip
make -j 10 install
```
