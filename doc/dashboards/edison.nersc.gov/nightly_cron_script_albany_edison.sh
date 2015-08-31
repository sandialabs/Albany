#!/bin/csh

cd /global/homes/i/ikalash/nightlyEdisonCDash

#module unload cmake netcdf-hdf5parallel/4.2.0 python
#module swap PrgEnv-pgi PrgEnv-gnu;
#module load cmake/3.1.3 python cray-netcdf-hdf5parallel usg-default-modules/1.2
#module load boost/1.57
#module load cmake

source edison_modules.sh 

#rm -rf /global/homes/i/ikalash/nightlyEdisonCDash/repos
#rm -rf /global/homes/i/ikalash/nightlyEdisonCDash/build
rm -rf /global/homes/i/ikalash/nightlyEdisonCDash/ctest_nightly.cmake.work
#rm -rf /global/homes/i/ikalash/nightlyEdisonCDash/nightly_log*
#rm -rf /global/homes/i/ikalash/nightlyEdisonCDash/results*

cat albanyFELIX ctest_nightly.cmake.frag >& ctest_nightly.cmake  

#export PATH=$PATH:/usr/lib64/openmpi/bin:/home/ikalash/Install/ParaView-4.3.1-Linux-64bit/bin:/home/ikalash/Install:/home/ikalash/Install/Cubit:/home/ikalash/Install/R2015a/bin:/home/ikalash/Desktop/nightlyAlbanyTests/Results/Trilinos/build/install

#export LD_LIBRARY_PATH=/usr/lib64:/usr/lib64/openmpi/lib

now=$(date +"%m_%d_%Y-%H_%M")
#LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now
LOG_FILE=/global/homes/i/ikalash/nightlyEdisonCDash/nightly_log_edisonAlbany.txt

eval "env  TEST_DIRECTORY=/global/homes/i/ikalash/nightlyEdisonCDash SCRIPT_DIRECTORY=/global/homes/i/ikalash/nightlyEdisonCDash ctest -VV -S /global/homes/i/ikalash/nightlyEdisonCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

cp -r build/AlbanyFELIXInstall/bin/* /project/projectdirs/piscees/nightlyEdisonCDashExe
chmod -R 0755 /project/projectdirs/piscees/nightlyEdisonCDashExe

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
