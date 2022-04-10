cmake_minimum_required (VERSION 2.8)

SET(CTEST_DO_SUBMIT "$ENV{DO_SUBMIT}")
SET(CTEST_TEST_TYPE "$ENV{TEST_TYPE}")
SET(CTEST_BUILD_OPTION "$ENV{BUILD_OPTION}")

if (1)
  # What to build and test
  IF(CTEST_BUILD_OPTION MATCHES "base-trilinos") 
    set (DOWNLOAD_TRILINOS TRUE)
    set (DOWNLOAD_ALBANY FALSE)
    set (BUILD_TRILINOS TRUE)
    set (BUILD_ALB64 FALSE) 
  ENDIF() 
  IF(CTEST_BUILD_OPTION MATCHES "base-albany") 
    set (DOWNLOAD_TRILINOS FALSE)
    set (DOWNLOAD_ALBANY TRUE)
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 TRUE) 
  ENDIF() 
  set (CLEAN_BUILD TRUE)
  set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
  IF(CTEST_BUILD_OPTION MATCHES "debug-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_TRILINOSDBG TRUE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_ALB64DBG FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_TRILINOSDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "debug-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_TRILINOSDBG FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_ALB64DBG TRUE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_ALB64DBG FALSE)
  ENDIF() 
  IF(CTEST_BUILD_OPTION MATCHES "clang-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG TRUE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_TRILINOSCLANG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clang-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG TRUE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_ALB64CLANG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clangdbg-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG TRUE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_TRILINOSCLANGDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "clangdbg-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG TRUE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_ALB64CLANGDBG FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "intel-trilinos")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS TRUE)
    set (BUILD_INTEL_ALBANY FALSE)
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_INTEL_TRILINOS FALSE)
  ENDIF()
  IF(CTEST_BUILD_OPTION MATCHES "intel-albany")
    set (BUILD_TRILINOS FALSE)
    set (BUILD_ALB64 FALSE)
    set (BUILD_TRILINOSCLANG FALSE)
    set (BUILD_ALB64CLANG FALSE)
    set (BUILD_TRILINOSCLANGDBG FALSE)
    set (BUILD_ALB64CLANGDBG FALSE)
    set (BUILD_INTEL_TRILINOS FALSE)
    set (BUILD_INTEL_ALBANY TRUE)
    set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#    set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
  ELSE()
    set (BUILD_INTEL_ALBANY FALSE)
  ENDIF()
else ()
  # This block is for testing. Set "if (1)" to "if (0)", and then freely mess
  # around with the settings in this block.

  # What to build and test
  set (DOWNLOAD_TRILINOS FALSE)
  set (DOWNLOAD_ALBANY FALSE)
  # See if we can get away with this for speed, at least until we get onto a
  # machine that can support a lengthy nightly.
  set (CLEAN_BUILD FALSE)
  set (BUILD_TRILINOS FALSE)
  set (BUILD_ALB64 FALSE)
  set (BUILD_TRILINOSCLANG FALSE)
  set (BUILD_ALB64CLANG FALSE)
  set (BUILD_TRILINOSCLANGDBG FALSE)
  set (BUILD_ALB64CLANGDBG FALSE)
  set (BUILD_INTEL_TRILINOS FALSE)
  set (BUILD_INTEL_ALBANY FALSE)
  set (CTEST_BUILD_CONFIGURATION  Release) # What type of build do you want ?
#  set (CTEST_BUILD_CONFIGURATION  Debug) # What type of build do you want ?
endif ()

set (extra_cxx_flags "")

find_program(UNAME NAMES uname)
macro(getuname name flag)
  exec_program("${UNAME}" ARGS "${flag}" OUTPUT_VARIABLE "${name}")
endmacro(getuname)

getuname(osname -s)
getuname(osrel  -r)
getuname(cpu    -m)

SET (MPI_BIN_DIR $ENV{MPI_BIN})
SET (MPI_LIB_DIR $ENV{MPI_LIB})

# Begin User inputs:
#set (CTEST_SITE "cee-compute011.sandia.gov" ) # generally the output of hostname
SITE_NAME(CTEST_SITE) # directly set CTEST_SITE to the output of `hostname`
set (CTEST_DASHBOARD_ROOT "$ENV{INSTALL_DIRECTORY}" ) # writable path
set (CTEST_SCRATCH_ROOT "$ENV{SCRATCH_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_ROOT "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
#set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CMAKE_GENERATOR "Ninja") 

set (CTEST_PROJECT_NAME "Albany" )
set (CTEST_SOURCE_NAME repos)
#set (CTEST_BUILD_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
#set (CTEST_BUILD_NAME "${osname}-${osrel}-${CTEST_BUILD_OPTION}-${CTEST_BUILD_CONFIGURATION}")
set (CTEST_BUILD_NAME "${osname}-${osrel}-${CTEST_BUILD_OPTION}")
set (CTEST_BINARY_NAME build)
set (CTEST_INSTALL_NAME test)

#  Over-write default limit for output posted to CDash site
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 5000000)
set(CTEST_CUSTOM_MAXIMUM_FAILED_TEST_OUTPUT_SIZE 5000000)

if (CTEST_BUILD_CONFIGURATION MATCHES "Debug")
# Runs tests longer if in debug mode
   set (CTEST_TEST_TIMEOUT 4200)
endif ()

set (PREFIX_DIR /projects/albany)

set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
# Build all results in a scratch space
set (CTEST_BINARY_DIRECTORY "${CTEST_SCRATCH_ROOT}/${CTEST_BINARY_NAME}")
# Trilinos, etc installed here
set (CTEST_INSTALL_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_INSTALL_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_INSTALL_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_INSTALL_DIRECTORY}")
endif ()

# Clean up storage area for nightly testing results
IF (CLEAN_BUILD)
  IF(EXISTS "${CTEST_BINARY_DIRECTORY}/Testing" )
    FILE(REMOVE_RECURSE "${CTEST_BINARY_DIRECTORY}/Testing")
  ENDIF()
ENDIF()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set (CTEST_CMAKE_COMMAND "${PREFIX_DIR}/bin/cmake")
set (CTEST_COMMAND "${PREFIX_DIR}/bin/ctest -D ${CTEST_TEST_TYPE}")
#set (CTEST_BUILD_FLAGS "-j16")
#IKT, 4/12/2022: the following is for Ninja build
set (CTEST_BUILD_FLAGS "${CTEST_BUILD_FLAGS}-k 999999")


set (CTEST_DROP_METHOD "https")

if (CTEST_DROP_METHOD STREQUAL "https")
  set (CTEST_DROP_SITE "sems-cdash-son.sandia.gov")
  set (CTEST_PROJECT_NAME "Albany")
  set (CTEST_DROP_LOCATION "/cdash/submit.php?project=Albany")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)
find_program (CTEST_SVN_COMMAND NAMES svn)

set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (Albany_REPOSITORY_LOCATION git@github.com:sandialabs/Albany.git)

if (DOWNLOAD_TRILINOS)
  #
  # Get the internal Trilinos repo
  #

  set (CTEST_CHECKOUT_COMMAND)

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone --branch develop ${Trilinos_REPOSITORY_LOCATION} ${CTEST_SOURCE_DIRECTORY}/Trilinos
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Trilinos repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")

ENDIF()

IF (DOWNLOAD_ALBANY) 
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

ENDIF ()

ctest_start(${CTEST_TEST_TYPE})

#
# Send the project structure to CDash
#

if (CTEST_DO_SUBMIT)
  ctest_submit (FILES "${CTEST_SCRIPT_DIRECTORY}/Project.xml"
    RETURN_VALUE HAD_ERROR)

  if (HAD_ERROR)
    message ("Cannot submit Albany Project.xml!")
  endif (HAD_ERROR)
endif (CTEST_DO_SUBMIT)

if (DOWNLOAD_TRILINOS)

  #
  # Update Trilinos
  #

  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Trilinos" RETURN_VALUE count)
  # assumes that we are already on the desired tracking branch, i.e.,
  # git checkout -b branch --track origin/branch
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message("Cannot update Trilinos!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Trilinos!")
  endif ()

ENDIF()

IF(DOWNLOAD_ALBANY) 
  #
  # Update Albany 
  #

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  CTEST_UPDATE(SOURCE "${CTEST_SOURCE_DIRECTORY}/Albany" RETURN_VALUE count)
  message("Found ${count} changed files")

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Update
      RETURN_VALUE  HAD_ERROR
      )

    if (HAD_ERROR)
      message("Cannot update Albany repository!")
    endif ()
  endif ()

  if (count LESS 0)
    message(FATAL_ERROR "Cannot update Albany!")
  endif ()

endif ()

INCLUDE(${CTEST_SCRIPT_DIRECTORY}/trilinos_macro.cmake)

if (BUILD_INTEL_TRILINOS)
  set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosIntelInstall")
  set (CONF_OPTS
     "-DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_LOCATION}"
     "-DCMAKE_BUILD_TYPE:STRING=RELEASE"
     "-DTPL_ENABLE_MPI:BOOL=ON"
     "-DMPI_BASE_DIR:PATH=$ENV{SEMS_OPENMPI_ROOT}"
     "-DCMAKE_CXX_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpicxx"
     "-DCMAKE_C_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpicc"
     "-DCMAKE_Fortran_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpifort"
     "-DTPL_ENABLE_Netcdf:BOOL=ON"
     "-DTPL_Netcdf_INCLUDE_DIRS:PATH=$ENV{SEMS_NETCDF_ROOT}/include"
     "-DNetcdf_LIBRARY_DIRS:PATH=$ENV{SEMS_NETCDF_ROOT}/lib"
     "-DTPL_ENABLE_HDF5:BOOL=OFF"
     "-DAmesos2_ENABLE_KLU2:BOOL=ON"
     "-DTPL_ENABLE_Boost:BOOL=ON"
     "-DTPL_ENABLE_BoostLib:BOOL=ON"
     "-DBoost_INCLUDE_DIRS:FILEPATH=$ENV{SEMS_BOOST_INCLUDE_PATH}"
     "-DBoost_LIBRARY_DIRS:FILEPATH=$ENV{SEMS_BOOST_LIBRARY_PATH}"
     "-DBoostLib_INCLUDE_DIRS:FILEPATH=$ENV{SEMS_BOOST_INCLUDE_PATH}"
     "-DBoostLib_LIBRARY_DIRS:FILEPATH=$ENV{SEMS_BOOST_LIBRARY_PATH}"
     "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
     "-DPiro_ENABLE_TESTS:BOOL=OFF"
     "-DRythmos_ENABLE_TESTS:BOOL=OFF"
     "-DROL_ENABLE_TESTS:BOOL=OFF"
     "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
     "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
     "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
     "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
     "-DTrilinos_ENABLE_Amesos:BOOL=ON"
     "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
     "-DTrilinos_ENABLE_Anasazi:BOOL=ON"
     "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
     "-DTrilinos_ENABLE_Belos:BOOL=ON"
     "-DTrilinos_ENABLE_COMPLEX_DOUBLE:BOOL=ON"
     "-DTrilinos_ENABLE_Epetra:BOOL=ON"
     "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
     "-DTrilinos_ENABLE_FEI:BOOL=OFF"
     "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
     "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
     "-DTrilinos_ENABLE_Intrepid:BOOL=ON"
     "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
     "-DKokkos_ENABLE_SERIAL:BOOL=ON"
     "-DKokkos_ENABLE_OPENMP:BOOL=OFF"
     "-DKokkos_ENABLE_PTHREAD:BOOL=OFF"
     "-DTrilinos_ENABLE_OpenMP:BOOL=OFF"
     "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
     "-DTrilinos_ENABLE_ML:BOOL=ON"
     "-DTrilinos_ENABLE_MueLu:BOOL=ON"
     "-DTrilinos_ENABLE_NOX:BOOL=ON"
     "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
     "-DTrilinos_ENABLE_PanzerExprEval:BOOL=ON"
     "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
     "-DTrilinos_ENABLE_Piro:BOOL=ON"
     "-DAnasazi_ENABLE_RBGen:BOOL=ON"
     "-DTrilinos_ENABLE_ROL:BOOL=ON"
     "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
     "-DTrilinos_ENABLE_Sacado:BOOL=ON"
     "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASAprepro_lib:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASConjoin:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASEjoin:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASEpu:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASAlgebra:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASExodiff:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
     "-DTrilinos_ENABLE_Shards:BOOL=ON"
     "-DTrilinos_ENABLE_ShyLU_DDFROSch:BOOL=ON"
     "-DTrilinos_ENABLE_STKUnit_tests:BOOL=ON"
     "-DTrilinos_ENABLE_STKIO:BOOL=ON"
     "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
     "-DTrilinos_ENABLE_STKExprEval:BOOL=ON"
     "-DTrilinos_ENABLE_Stokhos:BOOL=OFF"
     "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
     "-DTrilinos_ENABLE_Teko:BOOL=ON"
     "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
     "-DTrilinos_ENABLE_Thyra:BOOL=ON"
     "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
     "-DTrilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON"
     "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
     "-DTrilinos_ENABLE_TrilinosCouplings:BOOL=ON"
     "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
     "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
     "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
     "-DZoltan_ENABLE_ULONG_IDS:BOOL=OFF"
     "-DZOLTAN_BUILD_ZFDRIVE:BOOL=OFF"
     "-DTrilinos_ENABLE_DEBUG:BOOL=OFF"
     "-DPhalanx_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DStratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DTempus_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING='SERIAL'"
     "-DPhalanx_INDEX_SIZE_TYPE:STRING='INT'"
     "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
     "-DTrilinos_ENABLE_SCOREC:BOOL=OFF"
     "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
     "-DTpetra_INST_INT_INT:BOOL=OFF"
     "-DTpetra_INST_INT_LONG:BOOL=OFF"
     "-DTrilinos_ENABLE_Tempus:BOOL=ON"
     "-DTempus_ENABLE_TESTS:BOOL=OFF"
     "-DTempus_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTempus_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
     "-DTPL_Netcdf_PARALLEL:BOOL=ON"
     "-DTrilinos_ENABLE_CXX11:BOOL=ON"
     "-DTPL_FIND_SHARED_LIBS:BOOL=ON"
     "-DBUILD_SHARED_LIBS:BOOL=ON"
     "-DTrilinos_LINK_SEARCH_START_STATIC:BOOL=OFF"
     "-DMPI_EXEC=$ENV{SEMS_OPENMPI_ROOT}/bin/mpiexec"
     "-DPhalanx_ALLOW_MULTIPLE_EVALUATORS_FOR_SAME_FIELD:BOOL=ON"
     "-DTPL_ENABLE_Matio=OFF"
     "-DKOKKOS_ENABLE_LIBDL:BOOL=ON"
     "-DTrilinos_ENABLE_PanzerDofMgr:BOOL=ON"
     "-DTpetra_ENABLE_DEPRECATED_CODE=ON"
     "-DXpetra_ENABLE_DEPRECATED_CODE=ON"
  )
  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_trilinos("${CONF_OPTS}" "TrilinosIntel" "${INSTALL_LOCATION}")
  
endif (BUILD_INTEL_TRILINOS)

if (BUILD_TRILINOS OR BUILD_TRILINOSDBG)

  if (BUILD_TRILINOS) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstall")
    set(BTYPE "RELEASE") 
    set(CCFLAGS "-O3 -march=native -DNDEBUG -Wno-inconsistent-missing-override")
    set(CFLAGS "-O3 -march=native -DNDEBUG")
    set(FFLAGS "-O3 -march=native -DNDEBUG -Wa,-q")
  endif(BUILD_TRILINOS)
  if (BUILD_TRILINOSDBG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosDbg")
    set(BTYPE "DEBUG") 
    set(CCFLAGS "-g -O0 -Wno-inconsistent-missing-override") 
    set(CFLAGS "-g -O0")
    set(FFLAGS "-g -O0 -Wa,-q") 
  endif(BUILD_TRILINOSDBG)
  set (CONF_OPTS
    "-Wno-dev"
     "-DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_LOCATION}"
     "-DCMAKE_BUILD_TYPE:STRING=${BTYPE}"
     "-DTPL_ENABLE_MPI:BOOL=ON"
     "-DMPI_BASE_DIR:PATH=$ENV{SEMS_OPENMPI_ROOT}"
     "-DCMAKE_CXX_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpicxx"
     "-DCMAKE_C_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpicc"
     "-DCMAKE_Fortran_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpifort"
     "-DTPL_ENABLE_Netcdf:BOOL=ON"
     "-DTPL_Netcdf_INCLUDE_DIRS:PATH=$ENV{SEMS_NETCDF_ROOT}/include"
     "-DNetcdf_LIBRARY_DIRS:PATH=$ENV{SEMS_NETCDF_ROOT}/lib"
     "-DTPL_ENABLE_HDF5:BOOL=OFF"
     "-DAmesos2_ENABLE_KLU2:BOOL=ON"
     "-DTPL_ENABLE_Boost:BOOL=ON"
     "-DTPL_ENABLE_BoostLib:BOOL=ON"
     "-DBoost_INCLUDE_DIRS:FILEPATH=$ENV{SEMS_BOOST_INCLUDE_PATH}"
     "-DBoost_LIBRARY_DIRS:FILEPATH=$ENV{SEMS_BOOST_LIBRARY_PATH}"
     "-DBoostLib_INCLUDE_DIRS:FILEPATH=$ENV{SEMS_BOOST_INCLUDE_PATH}"
     "-DBoostLib_LIBRARY_DIRS:FILEPATH=$ENV{SEMS_BOOST_LIBRARY_PATH}"
     "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
     "-DPiro_ENABLE_TESTS:BOOL=OFF"
     "-DRythmos_ENABLE_TESTS:BOOL=OFF"
     "-DROL_ENABLE_TESTS:BOOL=OFF"
     "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
     "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
     "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
     "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
     "-DTrilinos_ENABLE_Amesos:BOOL=ON"
     "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
     "-DTrilinos_ENABLE_Anasazi:BOOL=ON"
     "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
     "-DTrilinos_ENABLE_Belos:BOOL=ON"
     "-DTrilinos_ENABLE_COMPLEX_DOUBLE:BOOL=ON"
     "-DTrilinos_ENABLE_Epetra:BOOL=ON"
     "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
     "-DTrilinos_ENABLE_FEI:BOOL=OFF"
     "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
     "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
     "-DTrilinos_ENABLE_Intrepid:BOOL=ON"
     "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
     "-DKokkos_ENABLE_SERIAL:BOOL=ON"
     "-DKokkos_ENABLE_OPENMP:BOOL=OFF"
     "-DKokkos_ENABLE_PTHREAD:BOOL=OFF"
     "-DTrilinos_ENABLE_OpenMP:BOOL=OFF"
     "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
     "-DTrilinos_ENABLE_ML:BOOL=ON"
     "-DTrilinos_ENABLE_MueLu:BOOL=ON"
     "-DTrilinos_ENABLE_NOX:BOOL=ON"
     "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
     "-DTrilinos_ENABLE_PanzerExprEval:BOOL=ON"
     "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
     "-DTrilinos_ENABLE_Piro:BOOL=ON"
     "-DAnasazi_ENABLE_RBGen:BOOL=ON"
     "-DTrilinos_ENABLE_ROL:BOOL=ON"
     "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
     "-DTrilinos_ENABLE_Sacado:BOOL=ON"
     "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASAprepro_lib:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASConjoin:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASEjoin:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASEpu:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASAlgebra:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASExodiff:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
     "-DTrilinos_ENABLE_Shards:BOOL=ON"
     "-DTrilinos_ENABLE_ShyLU_DDFROSch:BOOL=ON"
     "-DTrilinos_ENABLE_STKUnit_tests:BOOL=ON"
     "-DTrilinos_ENABLE_STKIO:BOOL=ON"
     "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
     "-DTrilinos_ENABLE_STKExprEval:BOOL=ON"
     "-DTrilinos_ENABLE_Stokhos:BOOL=OFF"
     "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
     "-DTrilinos_ENABLE_Teko:BOOL=ON"
     "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
     "-DTrilinos_ENABLE_Thyra:BOOL=ON"
     "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
     "-DTrilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON"
     "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
     "-DTrilinos_ENABLE_TrilinosCouplings:BOOL=ON"
     "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
     "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
     "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
     "-DZoltan_ENABLE_ULONG_IDS:BOOL=OFF"
     "-DZOLTAN_BUILD_ZFDRIVE:BOOL=OFF"
     "-DTrilinos_ENABLE_DEBUG:BOOL=OFF"
     "-DPhalanx_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DStratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DTempus_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING='SERIAL'"
     "-DPhalanx_INDEX_SIZE_TYPE:STRING='INT'"
     "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
     "-DTrilinos_ENABLE_SCOREC:BOOL=OFF"
     "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
     "-DTpetra_INST_INT_INT:BOOL=OFF"
     "-DTpetra_INST_INT_LONG:BOOL=OFF"
     "-DTrilinos_ENABLE_Tempus:BOOL=ON"
     "-DTempus_ENABLE_TESTS:BOOL=OFF"
     "-DTempus_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTempus_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
     "-DTPL_Netcdf_PARALLEL:BOOL=ON"
     "-DTrilinos_ENABLE_CXX11:BOOL=ON"
     "-DTPL_FIND_SHARED_LIBS:BOOL=ON"
     "-DBUILD_SHARED_LIBS:BOOL=ON"
     "-DTrilinos_LINK_SEARCH_START_STATIC:BOOL=OFF"
     "-DMPI_EXEC=$ENV{SEMS_OPENMPI_ROOT}/bin/mpiexec"
     "-DPhalanx_ALLOW_MULTIPLE_EVALUATORS_FOR_SAME_FIELD:BOOL=ON"
     "-DTPL_ENABLE_Matio=OFF"
     "-DKOKKOS_ENABLE_LIBDL:BOOL=ON"
     "-DTrilinos_ENABLE_PanzerDofMgr:BOOL=ON"
     "-DTpetra_ENABLE_DEPRECATED_CODE=ON"
     "-DXpetra_ENABLE_DEPRECATED_CODE=ON"
  )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_trilinos("${CONF_OPTS}" "Trilinos" "${INSTALL_LOCATION}")

endif (BUILD_TRILINOS OR BUILD_TRILINOSDBG)


if (BUILD_TRILINOSCLANG OR BUILD_TRILINOSCLANGDBG)

  if (BUILD_TRILINOSCLANG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11")
    set(BTYPE "RELEASE") 
    set(CCFLAGS "-O3 -march=native -DNDEBUG=1")
    set(CFLAGS "-O3 -march=native -DNDEBUG=1")
    set(FFLAGS "-O3 -march=native -DNDEBUG=1 -Wa,-q")
  endif (BUILD_TRILINOSCLANG) 
  if (BUILD_TRILINOSCLANGDBG) 
    set(INSTALL_LOCATION "${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11Dbg")
    set(BTYPE "DEBUG")
    set(CCFLAGS "-g -O0")
    set(CFLAGS "-g -O0")
    set(FFLAGS "-g -O0 -Wa,-q")
  endif (BUILD_TRILINOSCLANGDBG) 
  
  set (CONF_OPTS
    "-Wno-dev"
     "-DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_LOCATION}"
     "-DCMAKE_BUILD_TYPE:STRING=${BTYPE}"
     "-DTPL_ENABLE_MPI:BOOL=ON"
     "-DMPI_BASE_DIR:PATH=$ENV{SEMS_OPENMPI_ROOT}"
     "-DCMAKE_CXX_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpicxx"
     "-DCMAKE_C_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpicc"
     "-DCMAKE_Fortran_COMPILER:FILEPATH=$ENV{SEMS_OPENMPI_ROOT}/bin/mpifort"
     "-DCMAKE_CXX_FLAGS:STRING='-Wnoinconsistent-missing-override'"
     "-DTPL_ENABLE_Netcdf:BOOL=ON"
     "-DTPL_Netcdf_INCLUDE_DIRS:PATH=$ENV{SEMS_NETCDF_ROOT}/include"
     "-DNetcdf_LIBRARY_DIRS:PATH=$ENV{SEMS_NETCDF_ROOT}/lib"
     "-DTPL_ENABLE_HDF5:BOOL=OFF"
     "-DAmesos2_ENABLE_KLU2:BOOL=ON"
     "-DTPL_ENABLE_Boost:BOOL=ON"
     "-DTPL_ENABLE_BoostLib:BOOL=ON"
     "-DBoost_INCLUDE_DIRS:FILEPATH=$ENV{SEMS_BOOST_INCLUDE_PATH}"
     "-DBoost_LIBRARY_DIRS:FILEPATH=$ENV{SEMS_BOOST_LIBRARY_PATH}"
     "-DBoostLib_INCLUDE_DIRS:FILEPATH=$ENV{SEMS_BOOST_INCLUDE_PATH}"
     "-DBoostLib_LIBRARY_DIRS:FILEPATH=$ENV{SEMS_BOOST_LIBRARY_PATH}"
     "-DTrilinos_ENABLE_TESTS:BOOL=OFF"
     "-DPiro_ENABLE_TESTS:BOOL=OFF"
     "-DRythmos_ENABLE_TESTS:BOOL=OFF"
     "-DROL_ENABLE_TESTS:BOOL=OFF"
     "-DTrilinos_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTrilinos_ENABLE_ALL_PACKAGES:BOOL=OFF"
     "-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF"
     "-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF"
     "-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
     "-DTrilinos_ENABLE_Amesos:BOOL=ON"
     "-DTrilinos_ENABLE_Amesos2:BOOL=ON"
     "-DTrilinos_ENABLE_Anasazi:BOOL=ON"
     "-DTrilinos_ENABLE_AztecOO:BOOL=ON"
     "-DTrilinos_ENABLE_Belos:BOOL=ON"
     "-DTrilinos_ENABLE_COMPLEX_DOUBLE:BOOL=ON"
     "-DTrilinos_ENABLE_Epetra:BOOL=ON"
     "-DTrilinos_ENABLE_EpetraExt:BOOL=ON"
     "-DTrilinos_ENABLE_FEI:BOOL=OFF"
     "-DTrilinos_ENABLE_Ifpack:BOOL=ON"
     "-DTrilinos_ENABLE_Ifpack2:BOOL=ON"
     "-DTrilinos_ENABLE_Intrepid:BOOL=ON"
     "-DTrilinos_ENABLE_Intrepid2:BOOL=ON"
     "-DKokkos_ENABLE_SERIAL:BOOL=ON"
     "-DKokkos_ENABLE_OPENMP:BOOL=OFF"
     "-DKokkos_ENABLE_PTHREAD:BOOL=OFF"
     "-DTrilinos_ENABLE_OpenMP:BOOL=OFF"
     "-DTrilinos_ENABLE_MiniTensor:BOOL=ON"
     "-DTrilinos_ENABLE_ML:BOOL=ON"
     "-DTrilinos_ENABLE_MueLu:BOOL=ON"
     "-DTrilinos_ENABLE_NOX:BOOL=ON"
     "-DTrilinos_ENABLE_Pamgen:BOOL=ON"
     "-DTrilinos_ENABLE_PanzerExprEval:BOOL=ON"
     "-DTrilinos_ENABLE_Phalanx:BOOL=ON"
     "-DTrilinos_ENABLE_Piro:BOOL=ON"
     "-DAnasazi_ENABLE_RBGen:BOOL=ON"
     "-DTrilinos_ENABLE_ROL:BOOL=ON"
     "-DTrilinos_ENABLE_Rythmos:BOOL=ON"
     "-DTrilinos_ENABLE_Sacado:BOOL=ON"
     "-DTrilinos_ENABLE_SEACAS:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASExodus:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASAprepro_lib:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASConjoin:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASEjoin:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASEpu:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASAlgebra:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASExodiff:BOOL=ON"
     "-DTrilinos_ENABLE_SEACASIoss:BOOL=ON"
     "-DTrilinos_ENABLE_Shards:BOOL=ON"
     "-DTrilinos_ENABLE_ShyLU_DDFROSch:BOOL=ON"
     "-DTrilinos_ENABLE_STKUnit_tests:BOOL=ON"
     "-DTrilinos_ENABLE_STKIO:BOOL=ON"
     "-DTrilinos_ENABLE_STKMesh:BOOL=ON"
     "-DTrilinos_ENABLE_STKExprEval:BOOL=ON"
     "-DTrilinos_ENABLE_Stokhos:BOOL=OFF"
     "-DTrilinos_ENABLE_Stratimikos:BOOL=ON"
     "-DTrilinos_ENABLE_Teko:BOOL=ON"
     "-DTrilinos_ENABLE_Teuchos:BOOL=ON"
     "-DTrilinos_ENABLE_Thyra:BOOL=ON"
     "-DTrilinos_ENABLE_ThyraTpetraAdapters:BOOL=ON"
     "-DTrilinos_ENABLE_ThyraEpetraAdapters:BOOL=ON"
     "-DTrilinos_ENABLE_Tpetra:BOOL=ON"
     "-DTrilinos_ENABLE_TrilinosCouplings:BOOL=ON"
     "-DTrilinos_ENABLE_TriKota:BOOL=OFF"
     "-DTrilinos_ENABLE_Zoltan:BOOL=ON"
     "-DTrilinos_ENABLE_Zoltan2:BOOL=ON"
     "-DZoltan_ENABLE_ULONG_IDS:BOOL=OFF"
     "-DZOLTAN_BUILD_ZFDRIVE:BOOL=OFF"
     "-DTrilinos_ENABLE_DEBUG:BOOL=OFF"
     "-DPhalanx_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DStratimikos_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DTempus_ENABLE_TEUCHOS_TIME_MONITOR:BOOL=ON"
     "-DPhalanx_KOKKOS_DEVICE_TYPE:STRING='SERIAL'"
     "-DPhalanx_INDEX_SIZE_TYPE:STRING='INT'"
     "-DPhalanx_SHOW_DEPRECATED_WARNINGS:BOOL=OFF"
     "-DTrilinos_ENABLE_SCOREC:BOOL=OFF"
     "-DTpetra_INST_INT_LONG_LONG:BOOL=ON"
     "-DTpetra_INST_INT_INT:BOOL=OFF"
     "-DTpetra_INST_INT_LONG:BOOL=OFF"
     "-DTrilinos_ENABLE_Tempus:BOOL=ON"
     "-DTempus_ENABLE_TESTS:BOOL=OFF"
     "-DTempus_ENABLE_EXAMPLES:BOOL=OFF"
     "-DTempus_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON"
     "-DTPL_Netcdf_PARALLEL:BOOL=ON"
     "-DTrilinos_ENABLE_CXX11:BOOL=ON"
     "-DTPL_FIND_SHARED_LIBS:BOOL=ON"
     "-DBUILD_SHARED_LIBS:BOOL=ON"
     "-DTrilinos_LINK_SEARCH_START_STATIC:BOOL=OFF"
     "-DMPI_EXEC=$ENV{SEMS_OPENMPI_ROOT}/bin/mpiexec"
     "-DPhalanx_ALLOW_MULTIPLE_EVALUATORS_FOR_SAME_FIELD:BOOL=ON"
     "-DTPL_ENABLE_Matio=OFF"
     "-DKOKKOS_ENABLE_LIBDL:BOOL=ON"
     "-DTrilinos_ENABLE_PanzerDofMgr:BOOL=ON"
     "-DTpetra_ENABLE_DEPRECATED_CODE=ON"
     "-DXpetra_ENABLE_DEPRECATED_CODE=ON"
  )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
  do_trilinos("${CONF_OPTS}" "Trilinos" "${INSTALL_LOCATION}")

endif (BUILD_TRILINOSCLANG OR BUILD_TRILINOSCLANGDBG)

INCLUDE(${CTEST_SCRIPT_DIRECTORY}/albany_macro.cmake)

#
# Configure the Albany build using GO = long
#

if (BUILD_ALB64 OR BUILD_ALB64DBG OR BUILD_ALB64CLANG OR BUILD_ALB64CLANGDBG OR BUILD_INTEL_ALBANY)

if (BUILD_ALB64) 
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosInstall) 
  set(BUILDTYPE "RELEASE")
  set(FPE_CHECK "OFF")
  set(MESH_DEP_ON_SOLN "ON")
  set(MESH_DEP_ON_PARAMS "OFF")
endif(BUILD_ALB64) 
if (BUILD_INTEL_ALBANY)
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosIntelInstall)
  set(BUILDTYPE "RELEASE")
  set(FPE_CHECK "OFF")
  set(MESH_DEP_ON_SOLN "OFF")
  set(MESH_DEP_ON_PARAMS "ON")
endif (BUILD_INTEL_ALBANY)
if (BUILD_ALB64CLANG)
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11)
  set(BUILDTYPE "RELEASE")
  set(FPE_CHECK "OFF")
  set(MESH_DEP_ON_SOLN "OFF")
  set(MESH_DEP_ON_PARAMS "OFF")
endif (BUILD_ALB64CLANG)
if (BUILD_ALB64CLANGDBG)
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosInstallC11Dbg) 
  set(BUILDTYPE "DEBUG")
  set(FPE_CHECK "ON")
  set(MESH_DEP_ON_SOLN "OFF")
  set(MESH_DEP_ON_PARAMS "OFF")
endif (BUILD_ALB64CLANGDBG)
if (BUILD_ALB64DBG)
  set(TRILINSTALLDIR ${CTEST_INSTALL_DIRECTORY}/TrilinosDbg)
  set(BUILDTYPE "DEBUG")
  set(FPE_CHECK "ON")
  set(MESH_DEP_ON_SOLN "OFF")
  set(MESH_DEP_ON_PARAMS "OFF")
endif (BUILD_ALB64DBG)

  set (CONF_OPTIONS
    "-GNinja"
    "-DALBANY_TRILINOS_DIR:PATH=${TRILINSTALLDIR}"
#    "-DENABLE_ALBANY_EPETRA:BOOL=OFF"
    "-DENABLE_LANDICE:BOOL=ON"
    "-DENABLE_UNIT_TESTS:BOOL=ON"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=ON"
    "-DENABLE_MESH_DEPENDS_ON_SOLUTION:BOOL=${MESH_DEP_ON_SOLN}"
    "-DENABLE_MESH_DEPENDS_ON_PARAMETERS:BOOL=${MESH_DEP_ON_PARAMS}"
    "-DCMAKE_BUILD_TYPE:STRING=${BUILDTYPE}"
    "-DENABLE_STRONG_FPE_CHECK:BOOL=${FPE_CHECK}"
    )

  # First argument is the string of the configure options, second is the dashboard target (a name in a string)
if (BUILD_ALB64) 
  do_albany("${CONF_OPTIONS}" "Albany64Bit")
endif(BUILD_ALB64) 
if (BUILD_INTEL_ALBANY)
  do_albany("${CONF_OPTIONS}" "AlbanyIntel")
endif (BUILD_INTEL_ALBANY)
if (BUILD_ALB64CLANG)
  do_albany("${CONF_OPTIONS}" "Albany64BitClang")
endif (BUILD_ALB64CLANG)
if (BUILD_ALB64CLANGDBG)
  do_albany("${CONF_OPTIONS}" "Albany64BitClangDbg")
endif (BUILD_ALB64CLANGDBG)
if (BUILD_ALB64DBG)
  do_albany("${CONF_OPTIONS}" "Albany64BitDbg")
endif (BUILD_ALB64DBG)

endif (BUILD_ALB64 OR BUILD_ALB64DBG OR BUILD_ALB64CLANG OR BUILD_ALB64CLANGDBG OR BUILD_INTEL_ALBANY)


