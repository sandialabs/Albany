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
  ENDIF()
  MESSAGE("-- SEACAS_EPU = ${SEACAS_EPU}")
  MESSAGE("-- SEACAS_EXODIFF = ${SEACAS_EXODIFF}")
  MESSAGE("-- SEACAS_ALGEBRA = ${SEACAS_ALGEBRA}")
ENDIF()


# Paths to the actual executables
set(AlbanyPath                         ${Albany_BINARY_DIR}/src/Albany)
set(AlbanyTPath                        ${Albany_BINARY_DIR}/src/AlbanyT)
set(AlbanyTempusPath                   ${Albany_BINARY_DIR}/src/AlbanyTempus)
set(AlbanySGPath                       ${Albany_BINARY_DIR}/src/AlbanySG)
set(AlbanyDakotaPath                   ${Albany_BINARY_DIR}/src/AlbanyDakota)
set(AlbanyDakotaTPath                  ${Albany_BINARY_DIR}/src/AlbanyDakotaT)
set(AlbanyAnalysisPath                 ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
set(AlbanyAnalysisTPath                ${Albany_BINARY_DIR}/src/AlbanyAnalysisT)
set(AlbanyAdjointPath                  ${Albany_BINARY_DIR}/src/AlbanyAdjoint)
set(AlbanySGAdjointPath                ${Albany_BINARY_DIR}/src/AlbanySGAdjoint)
set(AlbanyCoupledPath                  ${Albany_BINARY_DIR}/src/AlbanyCoupled)
set(AlbanySGCoupledPath                ${Albany_BINARY_DIR}/src/AlbanySGCoupled)
set(AlbanyRBGenPath                    ${Albany_BINARY_DIR}/src/AlbanyRBGen)

IF (CISM_EXE_DIR)
set(CismAlbanyPath                ${CISM_EXE_DIR}/cism_driver)
ENDIF()


IF (ALBANY_MPI)
  set(SerialAlbany.exe                 ${SERIAL_CALL} ${AlbanyPath})
  set(SerialAlbanyT.exe                ${SERIAL_CALL} ${AlbanyTPath})
  set(SerialAlbanySG.exe               ${SERIAL_CALL} ${AlbanySGPath})
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
  IF (Kokkos_ENABLE_Cuda) 
    set(AlbanyT.exe                    ${PARALLEL_CALL} ${AlbanyTPath} ${KOKKOS_NDEVICES})
    set(AlbanyT8.exe                   ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST} ${AlbanyTPath} ${KOKKOS_NDEVICES})
    set(AlbanyAnalysisT.exe            ${PARALLEL_CALL} ${AlbanyAnalysisTPath} ${KOKKOS_NDEVICES})
    set(AlbanyDakotaT.exe              ${PARALLEL_CALL} ${AlbanyDakotaTPath} ${KOKKOS_NDEVICES})
    set(Albany.exe                     ${PARALLEL_CALL} ${AlbanyPath} ${KOKKOS_NDEVICES})
    set(Albany8.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST} ${AlbanyPath} ${KOKKOS_NDEVICES})
    set(AlbanyAnalysis.exe             ${PARALLEL_CALL} ${AlbanyAnalysisPath} ${KOKKOS_NDEVICES})
    set(AlbanyTempus.exe               ${PARALLEL_CALL} ${AlbanyTempusPath} ${KOKKOS_NDEVICES})
    set(AlbanyDakota.exe               ${PARALLEL_CALL} ${AlbanyDakotaPath} ${KOKKOS_NDEVICES})
    set(AlbanySG.exe                   ${PARALLEL_CALL} ${AlbanySGPath} ${KOKKOS_NDEVICES})
  ELSE() 
    set(AlbanyT.exe                    ${PARALLEL_CALL} ${AlbanyTPath})
    set(AlbanyT8.exe                   ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST} ${AlbanyTPath})
    set(AlbanyAnalysisT.exe            ${PARALLEL_CALL} ${AlbanyAnalysisTPath})
    set(AlbanyDakotaT.exe              ${PARALLEL_CALL} ${AlbanyDakotaTPath})
    set(Albany.exe                     ${PARALLEL_CALL} ${AlbanyPath})
    set(Albany8.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} ${MAX_MPI_RANKS} ${MPIPOST} ${AlbanyPath})
    set(AlbanyAnalysis.exe             ${PARALLEL_CALL} ${AlbanyAnalysisPath})
    set(AlbanyTempus.exe               ${PARALLEL_CALL} ${AlbanyTempusPath})
    set(AlbanyDakota.exe               ${PARALLEL_CALL} ${AlbanyDakotaPath})
    set(AlbanySG.exe                   ${PARALLEL_CALL} ${AlbanySGPath})
  ENDIF() 
ELSE()
  set(SerialAlbany.exe                 ${AlbanyPath})
  set(SerialAlbanyT.exe                ${AlbanyTPath})
  set(SerialAlbanySG.exe               ${AlbanySGPath})
  set(SerialAlbanyAnalysis.exe         ${AlbanyAnalysisPath})
  set(SerialAlbanyDakota.exe           ${AlbanyDakotaPath})
  set(Albany.exe                       ${AlbanyPath})
  set(AlbanyT.exe                      ${AlbanyTPath})
  set(AlbanyDakota.exe                 ${AlbanyDakotaPath})
  set(AlbanyDakotaT.exe                ${AlbanyDakotaTPath})
  set(AlbanySG.exe                     ${AlbanySGPath})
  set(AlbanyAnalysis.exe               ${AlbanyAnalysisPath})
  set(AlbanyAnalysisT.exe              ${AlbanyAnalysisTPath})
ENDIF()

# Only use 2 proc's for Coupled problems, since they system is 2x2
if (ALBANY_ASCR)
  IF (ALBANY_MPI)
    set(AlbanyAdjoint.exe   ${PARALLEL_CALL} ${AlbanyAdjointPath})
    set(AlbanySGAdjoint.exe ${PARALLEL_CALL} ${AlbanySGAdjointPath})
    set(AlbanyCoupled.exe   ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${MPIPOST} ${AlbanyCoupledPath})
    set(AlbanySGCoupled.exe ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${MPIPOST} ${AlbanySGCoupledPath})
  ELSE()
    set(AlbanyAdjoint.exe   ${AlbanyAdjointPath})
    set(AlbanySGAdjoing.exe ${AlbanySGAdjointPath})
    set(AlbanyCoupled.exe   ${AlbanyCoupledPath})
    set(AlbanySGCoupled.exe ${AlbanySGCoupledPath})
  ENDIF()
  set(SerialAlbanyCoupled.exe  ${AlbanyCoupledPath})
  set(SerialAlbanySGCoupled.exe  ${AlbanySGCoupledPath})
endif()

IF(ALBANY_LCM)
  set(PartitionTest.exe ${Albany_BINARY_DIR}/src/LCM/PartitionTest)
  set(PartitionTestT.exe ${Albany_BINARY_DIR}/src/LCM/PartitionTestT)
  set(Subdivision.exe   ${Albany_BINARY_DIR}/src/LCM/Subdivision)
  set(SubdivisionT.exe   ${Albany_BINARY_DIR}/src/LCM/SubdivisionT)
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
