#!/bin/sh

cd /home/ikalash/nightlyCDash

export PATH=$PATH:/usr/lib64/openmpi/bin:/home/ikalash/Install/ParaView-4.3.1-Linux-64bit/bin:/home/ikalash/Install:/home/ikalash/Install/Cubit:/home/ikalash/Install/R2015a/bin:/home/ikalash/nightlyAlbanyTests/Results/Trilinos/build-openmp/install

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib64/openmpi/lib
export OMP_NUM_THREADS=2
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export https_proxy="https://wwwproxy.ca.sandia.gov:80"
export http_proxy="http://wwwproxy.ca.sandia.gov:80"

now=$(date +"%m_%d_%Y-%H_%M")
#LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now
LOG_FILE=/home/ikalash/nightlyCDash/nightly_log_kokkosnode_openmp.txt

eval "env  TEST_DIRECTORY=/home/ikalash/nightlyCDash SCRIPT_DIRECTORY=/home/ikalash/nightlyCDash ctest -VV -S /home/ikalash/nightlyCDash/ctest_nightly_kokkosnode_openmp.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
