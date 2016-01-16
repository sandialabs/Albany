macro(do_albany)

# Configure the ALBANY build 
#
####################################################################################################################

SET_PROPERTY (GLOBAL PROPERTY SubProject Albany_CUVM)
SET_PROPERTY (GLOBAL PROPERTY Label Albany_CUVM)

SET(CONFIGURE_OPTIONS
  "-DALBANY_TRILINOS_DIR:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
  "-DNIGHTLY_TESTING:BOOL=ON"
  "-DENABLE_LCM:BOOL=ON"
  "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
  "-DENABLE_HYDRIDE:BOOL=OFF"
  "-DENABLE_SCOREC:BOOL=ON"
  "-DENABLE_SG:BOOL=OFF"
  "-DENABLE_ENSEMBLE:BOOL=OFF"
  "-DENABLE_FELIX:BOOL=ON"
  "-DENABLE_AERAS:BOOL=ON"
  "-DENABLE_QCAD:BOOL=OFF"
  "-DENABLE_MOR:BOOL=OFF"
  "-DENABLE_ATO:BOOL=OFF"
  "-DENABLE_ASCR:BOOL=OFF"
  "-DENABLE_CHECK_FPE:BOOL=OFF"
  "-DENABLE_LAME:BOOL=OFF"
  "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
  "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
#  "-DALBANY_MPI_OPTIONS:BOOL=ON"
#  "-DALBANY_MPI_EXEC:STRING=${MPI_BASE_DIR}/bin/mpiexec.hydra"
#  "-DALBANY_MPI_EXEC_NUMPROCS_FLAG:STRING=-n"
#  "-DALBANY_MPI_EXEC_MAX_NUMPROCS:STRING=4"
#  "-DALBANY_MPI_LEADING_OPTIONS:STRING=' -ppn 1 -env LD_LIBRARY_PATH /opt/intel/mkl/lib/intel64:/home/projects/x86-64-sandybridge-nvidia/cuda/7.5.7/lib64:/home/projects/gcc/4.9.0/lib64 '"
   )
 
if(NOT EXISTS "${CTEST_BINARY_DIRECTORY}/Albany")
  FILE(MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/Albany)
endif()

IF (CLEAN_BUILD)
# Initial cache info
set( CACHE_CONTENTS "
SITE:STRING=${CTEST_SITE}
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
BUILD_TESTING:BOOL=OFF
PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
" )
file(WRITE "${CTEST_BINARY_DIRECTORY}/Albany/CMakeCache.txt" "${CACHE_CONTENTS}")
ENDIF()


CTEST_CONFIGURE(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
          SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
          OPTIONS "${CONFIGURE_OPTIONS}"
          RETURN_VALUE HAD_ERROR
          APPEND
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Configure
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
    message(SEND_ERROR "Cannot submit Albany configure results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(SEND_ERROR "Cannot configure Albany build!")
endif()

#
# Build Albany
#
###################################################################################################################

SET(CTEST_BUILD_TARGET "Albany")

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
  IF(CTEST_DO_SUBMIT)
    CTEST_SUBMIT(PARTS Build
               RETURN_VALUE  S_HAD_ERROR
    )

    if(S_HAD_ERROR)
        message(SEND_ERROR "Cannot submit Albany build results!")
    endif()
  ENDIF()

  if(HAD_ERROR)
	message(SEND_ERROR "Cannot build Albany!")
  endif()

  message(SEND_ERROR "Encountered build errors in Albany build. Exiting!")

endif()

SET(CTEST_BUILD_TARGET "AlbanyT")

MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

CTEST_BUILD(
          BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
          RETURN_VALUE  HAD_ERROR
          NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
          APPEND
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Build
               RETURN_VALUE  S_HAD_ERROR
  )

  if(S_HAD_ERROR)
        message(SEND_ERROR "Cannot submit Albany build results!")
  endif()
ENDIF()

if(HAD_ERROR)
	message(SEND_ERROR "Cannot build AlbanyT!")
endif()

if(BUILD_LIBS_NUM_ERRORS GREATER 0)
    message(SEND_ERROR "Encountered build errors in AlbanyT build. Exiting!")
endif()

#
# Run Albany tests
#
##################################################################################################################

CTEST_TEST(
              BUILD "${CTEST_BINARY_DIRECTORY}/Albany"
#              INCLUDE "SCOREC_ThermoMechanicalCan_thermomech_tpetra"
#              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
              INCLUDE_LABEL "CUDA_TEST"
              #NUMBER_FAILED  TEST_NUM_FAILED
)

IF(CTEST_DO_SUBMIT)
  CTEST_SUBMIT(PARTS Test
               RETURN_VALUE  HAD_ERROR
  )

  if(HAD_ERROR)
    message(SEND_ERROR "Cannot submit Albany test results!")
  endif()
ENDIF()

endmacro(do_albany)
