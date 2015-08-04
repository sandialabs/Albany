
# Begin User inputs:
set (CTEST_SITE "camobap.ca.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_BUILD_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set (CTEST_BINARY_NAME build)

set (BOOSTDIR /home/ikalash/Install/boost_1_55_0)

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

  set_property (GLOBAL PROPERTY SubProject IKTAlbany32BitNoEpetra)
  set_property (GLOBAL PROPERTY Label IKTAlbany32BitNoEpetra)

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

if (BUILD_ALB64)

  # Configure the Albany 64 Bit build 
  # Builds everything!
  #

  set_property (GLOBAL PROPERTY SubProject IKTAlbany64Bit)
  set_property (GLOBAL PROPERTY Label IKTAlbany64BitNoEpetra)

  set (CISMDIR "/home/ikalash/Desktop/cism-piscees")
  set (TRILINSTALLDIR "/home/ikalash/Desktop/nightlyAlbanyTests/Results/Trilinos/build/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_GOAL:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON"
    "-DENABLE_MPAS_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_INTERFACE:BOOL=OFF"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CISMDIR}/libdycore"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DENABLE_LAME:BOOL=OFF")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbany64Bit")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbany64Bit)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany64Bit"
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

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany64Bit"
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
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany64Bit"
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

if (BUILD_ALB32)

  # Configure the Albany 32 Bit build 
  # Builds everything!
  #

  set_property (GLOBAL PROPERTY SubProject IKTAlbany32Bit)
  set_property (GLOBAL PROPERTY Label IKTAlbany32Bit)

  set (CISMDIR "/home/ikalash/Desktop/cism-piscees")
  set (TRILINSTALLDIR "/home/ikalash/Desktop/nightlyAlbanyTests/Results/Trilinos/build/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_GOAL:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON"
    "-DENABLE_MPAS_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_INTERFACE:BOOL=ON"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CISMDIR}/libdycore"
    "-DENABLE_LAME:BOOL=OFF")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbany32Bit")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbany32Bit)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany32Bit"
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

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany32Bit"
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
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany32Bit"
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


if (BUILD_ALB32_NOEPETRA)

  # Configure the Albany 32 Bit build 
  # Builds everything!
  #

  set_property (GLOBAL PROPERTY SubProject IKTAlbany32BitNoEpetra)
  set_property (GLOBAL PROPERTY Label IKTAlbany32BitNoEpetra)

  set (CISMDIR "/home/ikalash/Desktop/cism-piscees")
  set (TRILINSTALLDIR "/home/ikalash/Desktop/nightlyAlbanyTests/Results/Trilinos/build/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_GOAL:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON"
    "-DENABLE_MPAS_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_INTERFACE:BOOL=ON"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CISMDIR}/libdycore"
    "-DENABLE_LAME:BOOL=OFF")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbany32BitNoEpetra")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbany32BitNoEpetra)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany32BitNoEpetra"
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

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany32BitNoEpetra"
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
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany32BitNoEpetra"
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

#
# Configure the Albany build using GO = long
#

if (BUILD_ALB64_NOEPETRA)
  set_property (GLOBAL PROPERTY SubProject IKTAlbany64BitNoEpetra)
  set_property (GLOBAL PROPERTY Label IKTAlbany64BitNoEpetra)

  set (CISMDIR "/home/ikalash/Desktop/cism-piscees")
  set (TRILINSTALLDIR "/home/ikalash/Desktop/nightlyAlbanyTests/Results/Trilinos/build/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_GOAL:BOOL=OFF"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_CHECK_FPE:BOOL=ON"
    "-DENABLE_MPAS_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_INTERFACE:BOOL=OFF"
    "-DENABLE_64BIT_INT:BOOL=ON"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CISMDIR}/libdycore"
    "-DENABLE_LAME:BOOL=OFF")

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbany64BitNoEpetra")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbany64BitNoEpetra)
  endif ()

  #
  # The 64 bit build 
  #

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany64BitNoEpetra"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND
    )

  # Read the CTestCustom.cmake file to turn off ignored tests

  #CTEST_READ_CUSTOM_FILES("${CTEST_BINARY_DIRECTORY}/AlbanyT64")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany 64 bit configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot configure Albany 64 bit build!")
  endif ()

  #
  # Build Albany 64 bit
  #

  set (CTEST_BUILD_TARGET all)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany64BitNoEpetra"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany 64 bit build results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message(FATAL_ERROR "Cannot build Albany 64 bit!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message(FATAL_ERROR "Encountered build errors in Albany 64 bit build. Exiting!")
  endif ()
  #
  # Run Albany 64 bit tests
  #

  CTEST_TEST(
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbany64BitNoEpetra"
    #              PARALLEL_LEVEL "${CTEST_PARALLEL_LEVEL}"
    #              INCLUDE_LABEL "^${TRIBITS_PACKAGE}$"
    #NUMBER_FAILED  TEST_NUM_FAILED
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot submit Albany 64 bit test results!")
    endif ()
  endif ()
endif ()

# Add the path to Clang libraries needed for the Clang configure, build and sest cycle
#
# Need to add the openmpi libraries at the front of LD_LIBRARY_PATH
#

set (ENV{LD_LIBRARY_PATH} 
  /projects/albany/clang/lib:${INITIAL_LD_LIBRARY_PATH}
  )

if (BUILD_ALBFUNCTOR)
  # ALBANY_KOKKOS_UNDER_DEVELOPMENT build

  set_property (GLOBAL PROPERTY SubProject IKTAlbanyFunctor)
  set_property (GLOBAL PROPERTY Label IKTAlbanyFunctor)

  set (CISMDIR "/home/ikalash/Desktop/cism-piscees")
  set (TRILINSTALLDIR "/home/ikalash/Desktop/nightlyAlbanyTests/Results/Trilinos/build/install")

  set (CONFIGURE_OPTIONS
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
    "-DENABLE_LCM:BOOL=ON"
    "-DENABLE_CONTACT:BOOL=OFF"
    "-DENABLE_LCM_SPECULATIVE:BOOL=OFF"
    "-DENABLE_HYDRIDE:BOOL=ON"
    "-DENABLE_SG:BOOL=OFF"
    "-DENABLE_ENSEMBLE:BOOL=OFF"
    "-DENABLE_FELIX:BOOL=ON"
    "-DENABLE_AERAS:BOOL=ON"
    "-DENABLE_QCAD:BOOL=ON"
    "-DENABLE_MOR:BOOL=ON"
    "-DENABLE_ATO:BOOL=ON"
    "-DENABLE_ALBANY_EPETRA_EXE:BOOL=ON"
    "-DENABLE_AMP:BOOL=OFF"
    "-DENABLE_GOAL:BOOL=OFF"
    "-DENABLE_ASCR:BOOL=OFF"
    "-DENABLE_MPAS_INTERFACE:BOOL=ON"
    "-DENABLE_CISM_INTERFACE:BOOL=OFF"
    "-DCISM_INCLUDE_DIR:FILEPATH=${CISMDIR}/libdycore"
    "-DENABLE_KOKKOS_UNDER_DEVELOPMENT:BOOL=ON"
    "-DENABLE_64BIT_INT:BOOL=OFF"
    "-D ENABLE_CHECK_FPE:BOOL=ON"
    "-DENABLE_LAME:BOOL=OFF")
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctor")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctor)
  endif ()

  CTEST_CONFIGURE (
    BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctor"
    SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    APPEND)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure RETURN_VALUE S_HAD_ERROR)
    
    if (S_HAD_ERROR)
      message ("Cannot submit Albany configure results!")
      set (BUILD_ALBFUNCTOR FALSE)
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Albany build!")
    set (BUILD_ALBFUNCTOR FALSE)
  endif ()

  if (BUILD_ALBFUNCTOR)
    set (CTEST_BUILD_TARGET all)

    message ("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

    CTEST_BUILD (
      BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctor"
      RETURN_VALUE HAD_ERROR
      NUMBER_ERRORS BUILD_LIBS_NUM_ERRORS
      APPEND)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Build
        RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany build results!")
        set (BUILD_ALBFUNCTOR FALSE)
      endif ()
    endif ()

    if (HAD_ERROR)
      message ("Cannot build Albany!")
      set (BUILD_ALBFUNCTOR FALSE)
    endif ()

    if (BUILD_LIBS_NUM_ERRORS GREATER 0)
      message ("Encountered build errors in Albany build.")
      set (BUILD_ALBFUNCTOR FALSE)
    endif ()
  endif ()

  if (BUILD_ALBFUNCTOR)
    set (CTEST_TEST_TIMEOUT 120)
    CTEST_TEST (
      BUILD "${CTEST_BINARY_DIRECTORY}/IKTAlbanyFunctor"
      RETURN_VALUE HAD_ERROR)

    if (CTEST_DO_SUBMIT)
      ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

      if (S_HAD_ERROR)
        message ("Cannot submit Albany test results!")
      endif ()
    endif ()
  endif ()
endif ()
