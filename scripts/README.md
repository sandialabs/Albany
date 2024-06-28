# General configuration scripts for Trilinos/Albany

This folder contains scripts to configure Trilinos and Albany.
The scripts are organized in the following way

- do-cmake-XYZ.sh (XYZ=trilinos,albany): these are the main scripts
  you need to invoke to configure the projects. Inside, you can find
  some documentation on how to invoke them, and what the scripts
  expect you to set before calling them

- env-setup-files: this folder contains files you should source
  to get the correct environment for building on a certain machine
  with a certain compiler. E.g., `source env-setup-files/mappy-gny`
  will load the correct modules and set the correct env variables
  so that you can build trilinos/albany on mappy, using GNU compilers.

- kokkos-cache-files: this folder contains cache files that can be
  included inside other cache files to get the correct kokkos settings
  for the build you want. E.g., `kokkos-cache-files/device/openmp.cmake`
  will set the correct CMake variables to get OpenMP support.

- trilinos-cache-files: this folder contains cache files that can be
  used for the do-trilinos-cmake.sh script, to get the correct
  machine-specific settings. They are organized in machine-specific subfolders,
  since each machine may define different env variable for the TPLs,
  or use different compilers.

- albany-cache-files: this folder contains cache files that can be used
  for the do-albany-cmake.sh script, to get the correct settings for
  the desired Albany build. E.g., `albany-cache-files/albany-sfad-32.cmake`
  can be used to build all physics packages of Albany, usign Static FAD with
  length 32.

For instance, to build on mappy with GNU and OpenMP, the user can do the following

```bash
$ # Clone repos
$ mkdir trilinos && cd trilinos
$ git clone git@github.com:trilinos/trilinos source
$ cd ../ && mkdir albany && cd albany
$ git clone git@github.com:sandialabs/albany source
$ export SCRIPTS_DIR=$(pwd)/source/scripts
$ source ${SCRIPTS_DIR}/env-setup-files/mappy-gnu

$ # Install Trilinos
$ cd ../trilinos && mkdir build && cd build
$ export CACHE_FILE=${SCRIPTS_DIR}/trilinos-cache-files/mappy/trilinos-openmp.cmake
$ export SOURCE_DIR=$(pwd)/source
$ export INSTALL_DIR=$(pwd)/install
$ ${SCRIPTS_DIR}/do-cmake-trilinos.sh && make -j install

$ # Build Albany
$ cd ../../albany && mkdir build && cd build
$ export CACHE_FILE=${SCRIPTS_DIR}
$ export SOURCE_DIR=$(pwd)/source
$ export INSTALL_DIR=$(pwd)/install
$ export TRILINOS_DIR=$(pwd)/../../trilinos/install
$ ${SCRIPTS_DIR}/do-cmake-albany.sh && make -j
```
