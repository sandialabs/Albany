#!/bin/sh

cd /home/ikalash/Trilinos_Albany/nightlyCDash

rm -rf /home/ikalash/Trilinos_Albany/nightlyCDash/repos
rm -rf /home/ikalash/Trilinos_Albany/nightlyCDash/build
rm -rf /home/ikalash/Trilinos_Albany/nightlyCDash/ctest_nightly.cmake.work
rm -rf /home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log*
rm -rf /home/ikalash/Trilinos_Albany/nightlyCDash/results*

cat albanyKDVNoFunctor ctest_nightly.cmake.frag >& ctest_nightly.cmake  

export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64/atlas:/home/ikalash/Desktop/libmesh-0.7.1/libmesh/lib/x86_64-unknown-linux-gnu_opt:/home/ikalash/Desktop/libmesh-0.7.1/libmesh/contrib/lib/x86_64-unknown-linux-gnu_opt:/home/ikalash/Install/GSL/lib/:/home/ikalash/Install/netcdf-4.2-hack/lib:/home/ikalash/Install/boost_1_55_0/lib:/home/ikalash/Install/GSL/lib:/home/ikalash/Install/Python-2.7.6/lib:/home/ikalash/Install/Python-2.7.6/lib/python2.7/lib2to3:/home/ikalash/Install/Python-2.7.6/lib/python2.7/lib-dynload:/home/ikalash/Install/Python-2.7.6/lib/lib-tk:/usr/lib/x64-redhat-linux5E

PATH=/usr/local/bin:$PATH:/usr/lib/x64_64-redhat-linux5E/include:/usr/local/MATLAB/R2012a/bin/:/usr/local/tecplot360/bin/:/home/ikalash/Install/Cubit/:/home/ikalash/Install/ParaView-3.98.0-Linux-64bit/bin:/usr/local/bin:/home/ikalash/Install/GSL/include:/usr/lib64/atlas:/home/ikalash/Install/netcdf-4.2-hack/bin:/home/ikalash/Trilinos_Albany/Trilinos/seacas-build/install/bin:/usr/local/include:/home/ikalash/Install/Python-2.7.6/bin:/home/ikalash/Install/boost_1_55_0/include:/home/ikalash/Install/nco-4.4.0/bin:$NCARG_ROOT/bin:/home/ikalash/Trilinos_Albany/nightlyAlbanyTests/Trilinos_clean/seacas-build/install/bin
PATH=$PATH:/home/ikalash/Install/CEI/bin

export PATH

source /home/ikalash/paths

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=/home/ikalash/Trilinos_Albany/nightlyCDash/nightly_log_kdv_no_functor.txt

eval "env  TEST_DIRECTORY=/home/ikalash/Trilinos_Albany/nightlyCDash SCRIPT_DIRECTORY=/home/ikalash/Trilinos_Albany/nightlyCDash ctest -VV -S /home/ikalash/Trilinos_Albany/nightlyCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
