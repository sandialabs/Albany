#!/bin/sh

cd /home/ikalash/Desktop/nightlyCDash

rm -rf /home/ikalash/Desktop/nightlyCDash/repos
rm -rf /home/ikalash/Desktop/nightlyCDash/build
rm -rf /home/ikalash/Desktop/nightlyCDash/ctest_nightly.cmake.work
rm -rf /home/ikalash/Desktop/nightlyCDash/nightly_log.txt

export PATH=$PATH:/usr/lib64/openmpi/bin:/home/ikalash/Install/ParaView-4.3.1-Linux-64bit/bin:/home/ikalash/Install:/home/ikalash/Install/Cubit:/home/ikalash/Desktop/Trilinos/seacas-build/install/bin:/home/ikalash/Install/R2015a/bin

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib64/openmpi/lib

now=$(date +"%m_%d_%Y-%H_%M")
#LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now
LOG_FILE=/home/ikalash/Desktop/nightlyCDash/nightly_log.txt

eval "env  TEST_DIRECTORY=/home/ikalash/Desktop/nightlyCDash SCRIPT_DIRECTORY=/home/ikalash/Desktop/nightlyCDash ctest -VV -S /home/ikalash/Desktop/nightlyCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
