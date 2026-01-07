#!/bin/sh

cd /nightlyCDash

export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#export https_proxy="http://proxy.ca.sandia.gov:80"
#export http_proxy="http://proxy.ca.sandia.gov:80"
export PATH=$PATH:/tpls/install/ninja/build-cmake

LOG_FILE=/nightlyCDash/nightly_log_albany_openmp.txt

bash convert-cmake-to-cdash.sh openmp
bash create-new-cdash-cmake-script.sh openmp

eval "env  TEST_DIRECTORY=/nightlyCDash SCRIPT_DIRECTORY=/nightlyCDash ctest -VV -S /nightlyCDash/ctest_nightly_albanyOpenmp.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
