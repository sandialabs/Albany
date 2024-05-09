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

unset HTTPS_PROXY
unset HTTP_PROXY

export LM_LICENSE_FILE=7500@sitelicense.sandia.gov

if [ "${MODULESHOME:-}" = "" ]; then
  # Modules have not been set
  . /usr/share/Modules/init/bash
fi

if [ "$BUILD_OPT" = "intel-trilinos" ] || [ "$BUILD_OPT" = "intel-albany" ]; then

  echo "Intel release build" 
  source $SCRIPT_DIR/sems-intel-modules.sh >& $SCRATCH_DIR/sems-intel-modules.out 

elif [ "$BUILD_OPT" = "base-trilinos" ] || [ "$BUILD_OPT" = "base-albany" ] || [ "$BUILD_OPT" = "debug-trilinos" ] || [ "$BUILD_OPT" = "debug-albany" ]; then
  
  echo "Gcc build" 
  #gcc builds
  source $SCRIPT_DIR/sems-gcc-modules.sh >& $SCRATCH_DIR/sems-gcc-modules.out 

elif [ "$BUILD_OPT" = "clang-trilinos" ] || [ "$BUILD_OPT" = "clang-albany" ]; then

  echo "Clang release build" 
  # clang release builds
  source $SCRIPT_DIR/sems-clang-modules.sh >& $SCRATCH_DIR/sems-clang-modules.out

else  
  echo "Clang debug build" 
  # clang debug builds
  source $SCRIPT_DIR/sems-clang-modules.sh >& $SCRATCH_DIR/sems-clang-modules.out

fi


if [ ! -d "$INSTALL_DIR" ]; then
  /bin/mkdir $INSTALL_DIR
fi
if [ ! -d "$SCRATCH_DIR" ]; then
  /bin/mkdir $SCRATCH_DIR
fi

cd $SCRATCH_DIR

LOG_FILE=$SCRATCH_DIR/nightly_log_$BUILD_OPT.txt
export LD_LIBRARY_PATH==$LD_LIBRARY_PATH:$NETCDF_ROOT/lib:$PNETCDF_ROOT/lib 


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
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH" > ld_lib.path
eval "env BUILD_OPTION=$BUILD_OPT DO_SUBMIT=$SUBMIT_RESULTS TEST_TYPE=$THE_TEST_TYPE INSTALL_DIRECTORY=$INSTALL_DIR SCRATCH_DIRECTORY=$SCRATCH_DIR SCRIPT_DIRECTORY=$SCRIPT_DIR MPI_DIR=${SEMS_OPENMPI_ROOT} ctest -VV -S /projects/albany/nightlyAlbanyCDash/ctest_nightly.cmake" > $LOG_FILE 2>&1

# Copy a basic installation to /projects/albany for those who like a nightly
# build.
#cp -r build/TrilinosInstall/* /projects/albany/trilinos/nightly/;
#chmod -R a+X /projects/albany/trilinos/nightly;
#chmod -R a+r /projects/albany/trilinos/nightly;
