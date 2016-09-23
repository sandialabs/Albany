#!/bin/sh

cd /home/ikalash/Trilinos_Albany/nightlyCDash

cat albanyKDVFunctorSerial ctest_nightly.cmake.frag >& ctest_nightly.cmake  

export LD_LIBRARY_PATH=$SEMS_GCC_ROOT/lib:$SEMS_GCC_ROOT/lib64:$SEMS_OPENMPI_ROOT/lib:$SEMS_NETCDF_LIBRARY_PATH:$SEMS_HDF5_LIBRARY_PATH:$SEMS_BOOST_LIBRARY_PATH
echo "LD_LIBRARY_PATH = " $LD_LIBRARY_PATH

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log_kdv_functor_serial.txt

eval "env  TEST_DIRECTORY=/home/ikalash/Trilinos_Albany/nightlyCDash SCRIPT_DIRECTORY=/home/ikalash/Trilinos_Albany/nightlyCDash ctest -VV -S /home/ikalash/Trilinos_Albany/nightlyCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
