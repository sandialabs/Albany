## Clone repos and load modules
These build instructions are for compiling Albany Land Ice (ALI) on CEE ECW machines at Sandia National Laboratories.

Download Trilinos and Albany to the home directory by using `git`:
```sh
cd ${HOME}
git clone https://github.com/trilinos/Trilinos.git
git clone https://github.com/sandialabs/Albany.git
```
and load sems modules:
```sh
source cee_modules_clang.sh
```

## Building Trilinos/develop
Switch to the develop branch of Trilinos, make a build directory for Trilinos and copy over the configuration file provided:
```sh
cd ${HOME}/Trilinos
git checkout develop
mkdir build-clang-serial
cd build-clang-serial
cp ${HOME}/Albany/doc/LandIce/machines/cee/do-cmake-trilinos-cee-clang-serial .
```
Edit the configuration script to point to the repo location and install directory:
```sh
REPO_DIR=${HOME}/Trilinos
INSTALL_DIR=${HOME}/Trilinos/build-clang-serial/install
```

Build and install Trilinos by using the configuration file provided:
```sh
source do-cmake-trilinos-cee-clang-serial
make -j
make install
```
