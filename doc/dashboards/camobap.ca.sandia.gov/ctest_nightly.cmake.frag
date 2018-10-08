
# Begin User inputs:
set (CTEST_SITE "camobap.ca.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "fedora28-gcc8.1.1-${CTEST_BUILD_CONFIGURATION}")
set (CTEST_BINARY_NAME build)


set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "00:00:00 UTC")
set (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
set (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_BUILD_FLAGS "-j16")

set (CTEST_DROP_METHOD "http")

if (CTEST_DROP_METHOD STREQUAL "http")
  set (CTEST_DROP_SITE "cdash.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (Albany_REPOSITORY_LOCATION git@github.com:gahansen/Albany.git)
set (cism-piscees_REPOSITORY_LOCATION  git@github.com:E3SM-Project/cism-piscees.git)

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_BUILD_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  BUILD_TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${Albany_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()

if (DOWNLOAD)

  set (CTEST_CHECKOUT_COMMAND)
  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  #
  # Get Albany
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Albany")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Albany_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Albany
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Albany repository!")
    endif ()
  endif ()

  #
  # Get cism-piscees
  #
  #
  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/cism-piscees")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}"
      clone ${cism-piscees_REPOSITORY_LOCATION} -b ali_interface ${CTEST_SOURCE_DIRECTORY}/cism-piscees
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone cism-piscees repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")


endif ()

ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE  HAD_ERROR
    )

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot submit Albany Project.xml!")
  endif ()
endif ()

if (DOWNLOAD)

  #
  # Update Albany 
  #

  set_property (GLOBAL PROPERTY SubProject IKTAlbanyNoEpetra)
  set_property (GLOBAL PROPERTY Label IKTAlbanyNoEpetra)

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE count)
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot update Albany repository!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Albany!")
  endif ()

endif ()


if (BUILD_ALBANY)

  # Builds everything!
  #

  set_property (GLOBAL PROPERTY SubProject IKTAlbany)
  set_property (GLOBAL PROPERTY Label IKTAlbany)

  set (TRILINSTALLDIR "/home/ikalash/nightlyAlbanyTests/Results/Trilinos/build/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_TSUNAMI:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=OFF"
    "-DENABLE_MPAS_INTERFACE:BOOL=OFF"
    "-DENABLE_CISM_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_CHECK_COMPARISONS:BOOL=ON"
    "-DENABLE_CISM_EPETRA:BOOL=ON"
    "-DENABLE_CISM_REDUCED_COMM:BOOL=OFF"
    "-DSEACAS_EPU=/home/ikalash/Trilinos/seacas-build/install/bin/epu"
    "-DSEACAS_EXODIFF=/home/ikalash/Trilinos/seacas-build/install/bin/exodiff"
    "-DSEACAS_ALGEBRA=/home/ikalash/Trilinos/seacas-build/install/bin/algebra"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CTEST_SOURCE_DIRECTORY}/cism-piscees/libdycore"
    "-DINSTALL_ALBANY:BOOL=ON"
    "-DCMAKE_INSTALL_PREFIX:BOOL=${CTEST_BINARY_DIRECTORY}/IKTAlbanyInstall"
    "-DENABLE_PARAMETERS_DEPEND_ON_SOLUTION:BOOL=ON"
    "-DCISM_EXE_DIR:FILEPATH=${CTEST_BINARY_DIRECTORY}/IKTCismAlbany"
    "-DENABLE_USE_CISM_FLOW_PARAMETERS:BOOL=ON"
    "-DENABLE_LAME:BOOL=OFF")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbany")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbany)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure Albany build!")
  endif ()

  #
  # Build Albany
  #

  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build Albany!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in Albany build. Exiting!")
  endif ()

  #
  # Run Albany tests
  #
  
  set (CTEST_TEST_TIMEOUT 600)

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany test results!")
    endif ()
  endif ()

  #if (HAD_ERROR)
  #	message(FATAL_ERROR "Some Albany tests failed.")
  #endif ()

endif ()


if (BUILD_ALBANY_NOEPETRA)

  # Builds everything!
  #

  set_property (GLOBAL PROPERTY SubProject IKTAlbanyNoEpetra)
  set_property (GLOBAL PROPERTY Label IKTAlbanyNoEpetra)

  set (TRILINSTALLDIR "/home/ikalash/nightlyAlbanyTests/Results/Trilinos/build/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_TSUNAMI:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=OFF"
    "-DSEACAS_EPU=/home/ikalash/Trilinos/seacas-build/install/bin/epu"
    "-DSEACAS_EXODIFF=/home/ikalash/Trilinos/seacas-build/install/bin/exodiff"
    "-DSEACAS_ALGEBRA=/home/ikalash/Trilinos/seacas-build/install/bin/algebra"
    "-DENABLE_MPAS_INTERFACE:BOOL=OFF"
    "-DENABLE_CISM_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_CHECK_COMPARISONS:BOOL=ON"
    "-DENABLE_CISM_EPETRA:BOOL=OFF"
    "-DENABLE_CISM_REDUCED_COMM:BOOL=OFF"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CTEST_SOURCE_DIRECTORY}/cism-piscees/libdycore"
    "-DINSTALL_ALBANY:BOOL=ON"
    "-DCMAKE_INSTALL_PREFIX:BOOL=${CTEST_BINARY_DIRECTORY}/IKTAlbanyNoEpetraInstall"
    "-DENABLE_PARAMETERS_DEPEND_ON_SOLUTION:BOOL=ON"
    "-DENABLE_LAME:BOOL=OFF")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbanyNoEpetra")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbanyNoEpetra)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyNoEpetra"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure Albany build!")
  endif ()

  #
  # Build Albany
  #

  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyNoEpetra"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build Albany!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in Albany build. Exiting!")
  endif ()

  #
  # Run Albany tests
  #

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyNoEpetra"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany test results!")
    endif ()
  endif ()

  #if (HAD_ERROR)
  #	message(FATAL_ERROR "Some Albany tests failed.")
  #endif ()

endif ()

# Add the path to Clang libraries needed for the Clang configure, build and sest cycle
#
# Need to add the openmpi libraries at the front of LD_LIBRARY_PATH
#


if (BUILD_ALBFUNCTOR_OPENMP)
  # ALBANY_KOKKOS_UNDER_DEVELOPMENT build with OpenMP KokkosNode

  set_property (GLOBAL PROPERTY SubProject IKTAlbanyFunctorOpenMP)
  set_property (GLOBAL PROPERTY Label IKTAlbanyFunctorOpenMP)

  set (TRILINSTALLDIR "/home/ikalash/nightlyAlbanyTests/Results/Trilinos/build-openmp/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_LCM_TEST_EXES:BOOL=OFF"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_TSUNAMI:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_ATO:BOOL=OFF"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
    "-DSEACAS_EPU=/home/ikalash/Trilinos/seacas-build/install/bin/epu"
    "-DSEACAS_EXODIFF=/home/ikalash/Trilinos/seacas-build/install/bin/exodiff"
    "-DSEACAS_ALGEBRA=/home/ikalash/Trilinos/seacas-build/install/bin/algebra"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=OFF"
    "-DENABLE_MPAS_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_INTERFACE:BOOL=OFF"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CTEST_SOURCE_DIRECTORY}/cism-piscees/libdycore"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_DAKOTA_RESTART_EXAMPLES=OFF"
    "-DENABLE_SLFAD:BOOL=OFF"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-DALBANY_MPI_EXEC_TRAILING_OPTIONS='--map-by core'"
    "-DENABLE_LAME:BOOL=OFF")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP)
  endif ()

  CTEST_CONFIGURE (
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure RETURN_VALUE S_HAD_ERROR)
    
    if (S_HAD_ERROR)
      message ("Cannot submit Albany configure results!")
      set (BUILD_ALBFUNCTOR_OPENMP FALSE)
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
    set (BUILD_ALBFUNCTOR_OPENMP FALSE)
  endif ()

  if (BUILD_ALBFUNCTOR_OPENMP)
    set (CTEST_BUILD_TARGET all)

    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    CTEST_BUILD (
      BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP"
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany build results!")
        set (BUILD_ALBFUNCTOR_OPENMP FALSE)
      endif ()
    endif ()

    if (HAD_ERROR)
      message ("Cannot build Albany!")
      set (BUILD_ALBFUNCTOR_OPENMP FALSE)
    endif ()

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Albany build.")
      set (BUILD_ALBFUNCTOR_OPENMP FALSE)
    endif ()
  endif ()

  if (BUILD_ALBFUNCTOR_OPENMP)
    set (CTEST_TEST_TIMEOUT 1200)
    CTEST_TEST (
      BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctorOpenMP"
      RETURN_VALUE HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany test results!")
      endif ()
    endif ()
  endif ()
endif ()


if (BUILD_CISM_PISCEES)

  # Configure the CISM-Albany build 
  #
  set_property (GLOBAL PROPERTY SubProject IKTCismAlbany)
  set_property (GLOBAL PROPERTY Label IKTCismAlbany)

  set (TRILINSTALLDIR "/home/ikalash/nightlyAlbanyTests/Results/Trilinos/build/install")
  set (NETCDF_DIR /usr/local/netcdf-fortran-fedora28) 

  set (CONFIGURE_OPTIONS
    "-DCISM_USE_TRILINOS:BOOL=ON"
    "-DCISM_TRILINOS_DIR=${TRILINSTALLDIR}"
    "-DCISM_MPI_MODE:BOOL=ON"
    "-DCISM_SERIAL_MODE:BOOL=OFF"
    "-DCISM_BUILD_CISM_DRIVER:BOOL=ON"
    "-DALBANY_LANDICE_DYCORE:BOOL=ON"
    "-DALBANY_LANDICE_CTEST:BOOL=ON"
    "-DCISM_ALBANY_DIR=${CTEST_BINARY_DIRECTORY}/IKTAlbanyNoEpetraInstall"
    "-DCISM_NETCDF_DIR=${NETCDF_DIR}"
    "-DCISM_NETCDF_LIBS='netcdff'"
    "-DBUILD_SHARED_LIBS:BOOL=ON"
    "-DCMAKE_Fortran_FLAGS='-O2 -ffree-line-length-none -fPIC -fno-range-check'"
    "-DCMAKE_VERBOSE_MAKEFILE=OFF"
  )

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTCismAlbany")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTCismAlbany)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTCismAlbany"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/cism-piscees"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit CISM-Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure CISM-Albany build!")
  endif ()
 
   #
   # Build CISM-Albany
   #
   #
    set (CTEST_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTCismAlbany"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit CISM-Albany build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build CISM-Albany!")
  endif ()

  if (LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in CISM-Albany build. Exiting!")
  endif ()

  #
  # Run CISM-Albany tests
  #
  set (CTEST_TEST_TIMEOUT 1500)

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTCismAlbany"
#                  PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#                  INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
#    NUMBER_FAILED  TEST_NUM_FAILED
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit CISM-Albany test results!")
    endif ()
  endif ()

#  if (HAD_ERROR)
#  	message(FATAL_ERROR "Some CISM-Albany tests failed.")
#  endif ()

endif ()


if (BUILD_CISM_PISCEES_EPETRA)

  # Configure the CISM-Albany build 
  #
  set_property (GLOBAL PROPERTY SubProject IKTCismAlbanyEpetra)
  set_property (GLOBAL PROPERTY Label IKTCismAlbanyEpetra)

  set (TRILINSTALLDIR "/home/ikalash/nightlyAlbanyTests/Results/Trilinos/build/install")
  set (NETCDF_DIR /usr/local/netcdf-fortran-fedora28) 

  set (CONFIGURE_OPTIONS
    "-DCISM_USE_TRILINOS:BOOL=ON"
    "-DCISM_TRILINOS_DIR=${TRILINSTALLDIR}"
    "-DCISM_MPI_MODE:BOOL=ON"
    "-DCISM_SERIAL_MODE:BOOL=OFF"
    "-DCISM_BUILD_CISM_DRIVER:BOOL=ON"
    "-DALBANY_LANDICE_DYCORE:BOOL=ON"
    "-DALBANY_LANDICE_CTEST:BOOL=ON"
    "-DCISM_ALBANY_DIR=${CTEST_BINARY_DIRECTORY}/IKTAlbanyInstall"
    "-DCISM_NETCDF_DIR=${NETCDF_DIR}"
    "-DBUILD_SHARED_LIBS:BOOL=ON"
    "-DCISM_NETCDF_LIBS='netcdff'"
    "-DCMAKE_Fortran_FLAGS='-O2 -ffree-line-length-none -fPIC -fno-range-check'"
    "-DCMAKE_VERBOSE_MAKEFILE=OFF"
  )

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTCismAlbanyEpetra")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTCismAlbanyEpetra)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTCismAlbanyEpetra"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/cism-piscees"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit CISM-Albany configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure CISM-Albany build!")
  endif ()
 
   #
   # Build CISM-Albany
   #
   #
    set (CTEST_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTCismAlbanyEpetra"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit CISM-Albany build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build CISM-Albany!")
  endif ()

  if (LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in CISM-Albany build. Exiting!")
  endif ()

  #
  # Run CISM-Albany tests
  #

  set (CTEST_TEST_TIMEOUT 1500)
  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTCismAlbanyEpetra"
#                  PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
#                  INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
#    NUMBER_FAILED  TEST_NUM_FAILED
    RETURN_VALUE  HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit CISM-Albany test results!")
    endif ()
  endif ()

#  if (HAD_ERROR)
#  	message(FATAL_ERROR "Some CISM-Albany tests failed.")
#  endif ()

endif ()

