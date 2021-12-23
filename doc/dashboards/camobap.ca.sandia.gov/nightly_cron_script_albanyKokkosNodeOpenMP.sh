#!/bin/sh

cd /mnt/encrypted_sdc1/nightlyCDash

export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export https_proxy="http://proxy.ca.sandia.gov:80"
export http_proxy="http://proxy.ca.sandia.gov:80"

LOG_FILE=/mnt/encrypted_sdc1/nightlyCDash/nightly_log_kokkosnode_openmp.txt

eval "env  TEST_DIRECTORY=/mnt/encrypted_sdc1/nightlyCDash SCRIPT_DIRECTORY=/mnt/encrypted_sdc1/nightlyCDash ctest -VV -S /mnt/encrypted_sdc1/nightlyCDash/ctest_nightly_kokkosnode_openmp.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
