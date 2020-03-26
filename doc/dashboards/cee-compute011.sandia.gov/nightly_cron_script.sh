#!/bin/bash

SUBMIT_RESULTS=ON
THE_TEST_TYPE=Nightly
#SUBMIT_RESULTS=OFF
#THE_TEST_TYPE=Experimental

BUILD_OPT="$1"

if [ -z "$BUILD_OPT" ]; then

   echo "Please supply an argument: base-trilinos, base-albany, debug-trilinos, debug-albany, clang-trilinos, clang-albany, clangdbg-trilinos, clangdbg-albany, intel-trilinos or intel-albany"

   exit 1;

fi
SCRIPT_DIR=/projects/albany/nightlyAlbanyCDash
# Install directory holds final installed versions of the build. This is cross-mounted usually.
INSTALL_DIR=/projects/albany/nightlyAlbanyCDash
# The build area where the nightly products are built
SCRATCH_DIR=/scratch/albany

export LM_LICENSE_FILE=7500@sitelicense.sandia.gov

if [ "$BUILD_OPT" = "intel-trilinos" ] || [ "$BUILD_OPT" = "intel-albany" ]; then

  # Load a gcc as the intel compiler needs it
  if [ "${MODULESHOME:-}" = "" ]; then
    # Modules have not been set
    . /usr/share/Modules/init/bash
  fi
  module purge
  module load sierra-git/2.6.1
  module load sierra-devel/intel-18.0.3-intelmpi-5.1
  module load sparc-cmake

#   . /sierra/sntools/SDK/compilers/intel/composer_xe_2018.1.163/compilers_and_libraries/linux/bin/compilervars.sh intel64

   # Argh! The 2018.1.163 compiler install is apparently broken
#   export I_MPI_ROOT=/projects/sierra/linux_rh6/SDK/mpi/intel/5.1.2.150

elif [ "$BUILD_OPT" = "debug-trilinos" ] ||  [ "$BUILD_OPT" = "debug-albany" ]; then

  # Load latest gcc
  if [ "${MODULESHOME:-}" = "" ]; then
    # Modules have not been set
    . /usr/share/Modules/init/bash
  fi
  module purge
  module load sierra-git/2.6.1
#  module load sierra-devel/gcc-7.2.0-openmpi-1.10.2
  module load sierra-devel/gcc-8.1.0-openmpi-1.10.2
#  module load sierra-mkl/18.0-2018.1.163
  module load sierra-mkl/19.0-2019.0.117
  module load sparc-cmake

elif [ "$BUILD_OPT" = "clang-trilinos" ] || [ "$BUILD_OPT" = "clang-albany" ] || [ "$BUILD_OPT" = "clangdbg-trilinos" ] || [ "$BUILD_OPT" = "clangdbg-albany" ]; then

  # Need a gcc for stuff associated with clang
  if [ "${MODULESHOME:-}" = "" ]; then
    # Modules have not been set
    . /usr/share/Modules/init/bash
  fi

  module purge
  module load sparc-cmake 
  module load sierra-git/2.6.1
  module load sierra-devel/clang-7.0-openmpi-1.10.2
  #  module load sierra-mkl/18.0-2018.1.163
  module load sierra-mkl/19.0-2019.0.117
  #module unload sierra-compiler/clang/7.0
  module unload sierra-mpi/openmpi/1.10.2

#  export PATH=/projects/albany/bin:/projects/albany/trilinos/MPI_REL/bin:/projects/sierra/linux_rh6/SDK/compilers/clang/4.0-RHEL6/bin:/projects/sierra/linux_rh6/SDK/compilers/gcc/5.4.0-RHEL6/bin:/projects/sierra/linux_rh6/install/git/2.6.1/bin:/usr/bin:/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin

#  export LD_LIBRARY_PATH=/projects/sierra/linux_rh6/SDK/compilers/intel/composer_xe_2018.1.163/compilers_and_libraries/linux/mkl/lib/intel64:/projects/sierra/linux_rh6/SDK/compilers/clang/4.0-RHEL6/lib:/projects/sierra/linux_rh6/SDK/hwloc/lib:/projects/sierra/linux_rh6/SDK/compilers/gcc/5.4.0-RHEL6/lib64:/projects/sierra/linux_rh6/SDK/compilers/gcc/5.4.0-RHEL6/lib

else

  # Base gcc build
  if [ "${MODULESHOME:-}" = "" ]; then
    # Modules have not been set
    . /usr/share/Modules/init/bash
  fi
  module purge
  module load sierra-git/2.6.1
#  module load sierra-compiler/gcc/5.2.0
  module load sierra-devel/gcc-8.1.0-openmpi-1.10.2
#  module load sierra-mkl/18.0-2018.1.163
  module load sierra-mkl/19.0-2019.0.117
  module load sparc-cmake

#  export PATH=/projects/albany/bin:/projects/albany/trilinos/MPI_REL/bin:/projects/sierra/linux_rh6/SDK/compilers/clang/4.0-RHEL6/bin:/projects/sierra/linux_rh6/SDK/mpi/openmpi/1.10.2-gcc-5.4.0-RHEL6/bin:/projects/sierra/linux_rh6/SDK/compilers/gcc/5.4.0-RHEL6/bin:/projects/sierra/linux_rh6/install/git/2.6.1/bin:/usr/bin:/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin

#  export LD_LIBRARY_PATH=/projects/sierra/linux_rh6/SDK/compilers/intel/composer_xe_2018.1.163/compilers_and_libraries/linux/mkl/lib/intel64:/projects/sierra/linux_rh6/SDK/compilers/clang/4.0-RHEL6/lib:/projects/sierra/linux_rh6/SDK/hwloc/lib:/projects/sierra/linux_rh6/SDK/mpi/openmpi/1.10.2-gcc-5.4.0-RHEL6/lib:/projects/sierra/linux_rh6/SDK/compilers/gcc/5.4.0-RHEL6/lib64:/projects/sierra/linux_rh6/SDK/compilers/gcc/5.4.0-RHEL6/lib

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

eval "env BUILD_OPTION=$BUILD_OPT DO_SUBMIT=$SUBMIT_RESULTS TEST_TYPE=$THE_TEST_TYPE INSTALL_DIRECTORY=$INSTALL_DIR SCRATCH_DIRECTORY=$SCRATCH_DIR SCRIPT_DIRECTORY=$SCRIPT_DIR ctest -VV -S /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
