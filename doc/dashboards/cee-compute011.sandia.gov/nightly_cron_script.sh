#!/bin/sh

cd /projects/AppComp/nightly_gahanse/cee-compute011

export http_proxy=http://wwwproxy.sandia.gov:80
export https_proxy=http://wwwproxy.sandia.gov:80

export PATH=/projects/albany/bin:/projects/albany/trilinos/MPI_REL/bin:/sierra/sntools/SDK/compilers/clang/3.7-RHEL6/bin:/sierra/sntools/SDK/compilers/clang/3.7-RHEL6/bin/scan-build:/sierra/sntools/SDK/compilers/clang/3.7-RHEL6/bin/scan-view:/sierra/sntools/SDK/mpi/openmpi/1.8.8-gcc-5.2.0-RHEL6/bin:/projects/sierra/linux_rh6/install/SNTOOLS_dir/master/sntools/job_scripts/linux_rh6/openmpi:/sierra/sntools/SDK/compilers/gcc/5.2.0-RHEL6/bin:/projects/sierra/linux_rh6/install/git/2.6.1/bin:/projects/sierra/linux_rh6/install/git/bin:/usr/bin:/bin:/sbin:/usr/sbin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin

export LD_LIBRARY_PATH=/sierra/sntools/SDK/compilers/intel/composer_xe_2016.3.210/mkl/lib/intel64:/sierra/sntools/SDK/compilers/clang/3.7-RHEL6/lib:/sierra/sntools/SDK/hwloc/lib:/sierra/sntools/SDK/mpi/openmpi/1.8.8-gcc-5.2.0-RHEL6/lib:/sierra/sntools/SDK/compilers/gcc/5.2.0-RHEL6/lib64:/sierra/sntools/SDK/compilers/gcc/5.2.0-RHEL6/lib:/projects/albany/lib

now=$(date +"%m_%d_%Y-%H_%M")
#LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now
LOG_FILE=/projects/AppComp/nightly_gahanse/cee-compute011/nightly_log.txt

# I want to run incremental builds to reduce the length of the nightly; with the
# new Intel builds, we're up to ~12 hours. Make sure the CMake files are
# completely zorched so we always do a clean configure. (I don't see an option
# for ctest_configure that I trust will do a configuration completely from
# scratch.)

for i in `ls build/`; do
    rm -f build/$i/CMakeCache.txt
    rm -rf build/$i/CMakeFiles
done

eval "env TEST_DIRECTORY=/projects/AppComp/nightly_gahanse/cee-compute011 SCRIPT_DIRECTORY=/projects/AppComp/nightly_gahanse/cee-compute011 /projects/albany/bin/ctest -VV -S /ascldap/users/gahanse/Codes/Albany/doc/dashboards/cee-compute011.sandia.gov/ctest_nightly.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
