#!/bin/bash

SUBMIT_RESULTS=ON
#SUBMIT_RESULTS=OFF
THE_TEST_TYPE=Nightly
#THE_TEST_TYPE=Experimental

if [ "${MODULESHOME:-}" = "" ]; then
  # Modules have not been set
  . /usr/share/Modules/init/bash
  module purge
  module load sierra-git/2.6.1
  module load sierra-compiler/gcc/5.2.0
  module load sierra-mkl/17.0-2017.2.174
else
  module purge
  module load sierra-git/2.6.1
  module load sierra-compiler/gcc/5.2.0
  module load sierra-mkl/17.0-2017.2.174
fi

BUILD_OPT="$1"

if [ -z "$BUILD_OPT" ]; then

   echo "Please supply an argument: base, clang, or intel"

   exit 1;

fi

SCRIPT_DIR=/ascldap/users/gahanse/Codes/Albany/doc/dashboards/cee-compute011.sandia.gov
# Install directory holds final installed versions of the build. This is cross-mounted usually.
INSTALL_DIR=/projects/AppComp/nightly_gahanse/cee-compute011
# The build area where the nightly products are built
SCRATCH_DIR=/scratch/gahanse

export LM_LICENSE_FILE=7500@sitelicense.sandia.gov

if [ "$BUILD_OPT" = "intel" ]; then
   . /sierra/sntools/SDK/compilers/intel/composer_xe_2017.2.174/compilers_and_libraries/linux/bin/compilervars.sh intel64

else

export PATH=/projects/albany/bin:/projects/albany/trilinos/MPI_REL/bin:/sierra/sntools/SDK/compilers/clang/3.7-RHEL6/bin:/sierra/sntools/SDK/mpi/openmpi/1.8.8-gcc-5.2.0-RHEL6/bin:/sierra/sntools/SDK/compilers/gcc/5.2.0-RHEL6/bin:/projects/sierra/linux_rh6/install/git/2.6.1/bin:/projects/sierra/linux_rh6/install/git/bin:/usr/bin:/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin

export LD_LIBRARY_PATH=/sierra/sntools/SDK/compilers/intel/composer_xe_2017.2.174/compilers_and_libraries/linux/mkl/lib/intel64:/sierra/sntools/SDK/compilers/clang/3.7-RHEL6/lib:/sierra/sntools/SDK/hwloc/lib:/sierra/sntools/SDK/mpi/openmpi/1.8.8-gcc-5.2.0-RHEL6/lib:/sierra/sntools/SDK/compilers/gcc/5.2.0-RHEL6/lib64:/sierra/sntools/SDK/compilers/gcc/5.2.0-RHEL6/lib:/projects/albany/lib
fi


if [ ! -d "$INSTALL_DIR" ]; then
  /bin/mkdir $INSTALL_DIR
fi
if [ ! -d "$SCRATCH_DIR" ]; then
  /bin/mkdir $SCRATCH_DIR
fi

cd $SCRATCH_DIR

now=$(date +"%m_%d_%Y-%H_%M")
#LOG_FILE=/projects/AppComp/nightly/cee-compute011/nightly_$now
LOG_FILE=$SCRATCH_DIR/nightly_log_$BUILD_OPT.txt

# I want to run incremental builds to reduce the length of the nightly; with the
# new Intel builds, we're up to ~12 hours. Make sure the CMake files are
# completely zorched so we always do a clean configure. (I don't see an option
# for ctest_configure that I trust will do a configuration completely from
# scratch.)

#for i in `ls build/`; do
#    rm -f build/$i/CMakeCache.txt
#    rm -rf build/$i/CMakeFiles
#done

echo "Date and time is $now" > $LOG_FILE

eval "env BUILD_OPTION=$BUILD_OPT DO_SUBMIT=$SUBMIT_RESULTS TEST_TYPE=$THE_TEST_TYPE INSTALL_DIRECTORY=$INSTALL_DIR SCRATCH_DIRECTORY=$SCRATCH_DIR SCRIPT_DIRECTORY=$SCRIPT_DIR /projects/albany/bin/ctest -VV -S /ascldap/users/gahanse/Codes/Albany/doc/dashboards/cee-compute011.sandia.gov/ctest_nightly.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
