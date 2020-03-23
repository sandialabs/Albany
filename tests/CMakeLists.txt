##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Setup paths for SEACAS tools needed to run some of the tests
IF (ALBANY_SEACAS)
  IF(DEFINED ALBANY_SEACAS_PATH)
    find_program(
         SEACAS_EPU
         NAMES epu
         PATHS ${ALBANY_SEACAS_PATH}
    )
    find_program(
         SEACAS_EXODIFF
         NAMES exodiff
         PATHS ${ALBANY_SEACAS_PATH}
    )
    find_program(
         SEACAS_ALGEBRA
         NAMES algebra
         PATHS ${ALBANY_SEACAS_PATH}
    )
    find_program(
         SEACAS_DECOMP
         NAMES decomp
         PATHS ${ALBANY_SEACAS_PATH}
    )
  ELSE()
    find_program(
         SEACAS_EPU
         NAMES epu
         PATHS ${Trilinos_BIN_DIRS} ENV PATH
    )
    find_program(
         SEACAS_EXODIFF
         NAMES exodiff
         PATHS ${Trilinos_BIN_DIRS} ENV PATH
    )
    find_program(
         SEACAS_ALGEBRA
         NAMES algebra
         PATHS ${Trilinos_BIN_DIRS} ENV PATH
    )
    find_program(
         SEACAS_DECOMP
         NAMES decomp
         PATHS ${Trilinos_BIN_DIRS} ENV PATH
    )
  ENDIF()
  MESSAGE("-- SEACAS_ALGEBRA = ${SEACAS_ALGEBRA}")
  MESSAGE("-- SEACAS_DECOMP  = ${SEACAS_DECOMP}")
  MESSAGE("-- SEACAS_EPU     = ${SEACAS_EPU}")
  MESSAGE("-- SEACAS_EXODIFF = ${SEACAS_EXODIFF}")
ENDIF()


# Paths to the actual executables
set(AlbanyPath                         ${Albany_BINARY_DIR}/src/Albany)
set(AlbanyDakotaPath                   ${Albany_BINARY_DIR}/src/AlbanyDakota)
set(AlbanyAnalysisPath                 ${Albany_BINARY_DIR}/src/AlbanyAnalysis)

IF (CISM_EXE_DIR)
  set(CismAlbanyPath                ${CISM_EXE_DIR}/cism_driver)
ENDIF()

IF (ALBANY_MPI)
  set(SerialAlbany.exe                 ${SERIAL_CALL} ${AlbanyPath})
  set(SerialAlbanyAnalysis.exe         ${SERIAL_CALL} ${AlbanyAnalysisPath})
  set(SerialAlbanyDakota.exe           ${SERIAL_CALL} ${AlbanyDakotaPath})
  # Do not test on greater than Trilinos_MPI_EXEC_MAX_NUMPROCS configured in Trilinos build
  # or explicity given at Albany configure time -D ALBANY_MPI_EXEC_MAX_NUMPROCS
  IF(DEFINED MPIMNP)
    IF(MPIMNP LESS 8)
      SET(MAX_MPI_RANKS ${MPIMNP})
    ELSE(MPIMNP LESS 8)
      SET(MAX_MPI_RANKS 8)
    ENDIF(MPIMNP LESS 8)
  ELSE(DEFINED MPIMNP)
      SET(MAX_MPI_RANKS 4)
  ENDIF(DEFINED MPIMNP)
  IF (ALBANY_ENABLE_CUDA) 
    set(Albany.exe                     ${PARALLEL_CALL} ${AlbanyPath} ${KOKKOS_NDEVICES})
    set(Albany8.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST} ${AlbanyPath} ${KOKKOS_NDEVICES})
    set(AlbanyAnalysis.exe             ${PARALLEL_CALL} ${AlbanyAnalysisPath} ${KOKKOS_NDEVICES})
    set(AlbanyDakota.exe               ${PARALLEL_CALL} ${AlbanyDakotaPath} ${KOKKOS_NDEVICES})
  ELSE() 
    set(Albany.exe                     ${PARALLEL_CALL} ${AlbanyPath})
    set(Albany8.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST} ${AlbanyPath})
    set(AlbanyAnalysis.exe             ${PARALLEL_CALL} ${AlbanyAnalysisPath})
    set(AlbanyDakota.exe               ${PARALLEL_CALL} ${AlbanyDakotaPath})
  ENDIF() 
ELSE()
  set(SerialAlbany.exe                 ${AlbanyPath})
  set(SerialAlbanyAnalysis.exe         ${AlbanyAnalysisPath})
  set(SerialAlbanyDakota.exe           ${AlbanyDakotaPath})
  set(Albany.exe                       ${AlbanyPath})
  set(AlbanyDakota.exe                 ${AlbanyDakotaPath})
  set(AlbanyAnalysis.exe               ${AlbanyAnalysisPath})
ENDIF()

IF(ALBANY_LCM)
  set(PartitionTest.exe ${Albany_BINARY_DIR}/src/LCM/PartitionTest)
  set(Subdivision.exe   ${Albany_BINARY_DIR}/src/LCM/Subdivision)
  set(MPS.exe           ${Albany_BINARY_DIR}/src/LCM/MaterialPointSimulator)
  set(MPST.exe           ${Albany_BINARY_DIR}/src/LCM/MaterialPointSimulatorT)
  set(DTK_Interp_and_Error.exe ${Albany_BINARY_DIR}/src/LCM/DTK_Interp_and_Error)
  set(DTK_Interp_Volume_to_NS.exe ${Albany_BINARY_DIR}/src/LCM/DTK_Interp_Volume_to_NS)
  IF (ALBANY_MPI)
    set(Parallel_DTK_Interp_and_Error.exe ${PARALLEL_CALL} ${DTK_Interp_and_Error.exe})
    set(Parallel_DTK_Interp_Volume_to_NS.exe ${PARALLEL_CALL} ${DTK_Interp_Volume_to_NS.exe})
  ENDIF()
ENDIF(ALBANY_LCM)

add_subdirectory(small)
IF (ALBANY_LARGE_TESTS)
  add_subdirectory(large)
ENDIF()

IF (ALBANY_UNIT_TESTS)
  add_subdirectory(unit)
ENDIF()
