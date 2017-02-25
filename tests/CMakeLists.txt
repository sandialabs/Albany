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

IF (CISM_EXE_DIR)
set(CismAlbanyPath                ${CISM_EXE_DIR}/cism_driver)
ENDIF()


IF (ALBANY_MPI)
  set(SerialAlbany.exe                 ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanyPath})
  set(SerialAlbanyT.exe                ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanyTPath})
  set(SerialAlbanySG.exe               ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanySGPath})
  set(SerialAlbanyAnalysis.exe         ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanyAnalysisPath})
  set(SerialAlbanyDakota.exe           ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanyDakotaPath})
  IF (Kokkos_ENABLE_Cuda) 
    set(AlbanyT.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyTPath} ${KOKKOS_NDEVICES})
    set(AlbanyT8.exe                   ${MPIEX} ${MPIPRE} ${MPINPF} 8 ${MPIPOST} ${AlbanyTPath} ${KOKKOS_NDEVICES})
    set(AlbanyAnalysisT.exe            ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyAnalysisTPath} ${KOKKOS_NDEVICES})
    set(AlbanyDakotaT.exe              ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyDakotaTPath} ${KOKKOS_NDEVICES})
    set(Albany.exe                     ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyPath})
    set(Albany8.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} 8 ${MPIPOST} ${AlbanyPath})
    set(AlbanyAnalysis.exe             ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyAnalysisPath})
    set(AlbanyTempus.exe               ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyTempusPath})
    set(AlbanyDakota.exe               ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyDakotaPath})
    set(AlbanySG.exe                   ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanySGPath})
  ELSE() 
    set(AlbanyT.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyTPath})
    set(AlbanyT8.exe                   ${MPIEX} ${MPIPRE} ${MPINPF} 8 ${MPIPOST} ${AlbanyTPath})
    set(AlbanyAnalysisT.exe            ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyAnalysisTPath})
    set(AlbanyDakotaT.exe              ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyDakotaTPath})
    set(Albany.exe                     ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyPath})
    set(Albany8.exe                    ${MPIEX} ${MPIPRE} ${MPINPF} 8 ${MPIPOST} ${AlbanyPath})
    set(AlbanyAnalysis.exe             ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyAnalysisPath})
    set(AlbanyTempus.exe               ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyTempusPath})
    set(AlbanyDakota.exe               ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyDakotaPath})
    set(AlbanySG.exe                   ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanySGPath})
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
    set(AlbanyAdjoint.exe   ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyAdjointPath})
    set(AlbanySGAdjoint.exe ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanySGAdjointPath})
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
ENDIF(ALBANY_LCM)

add_subdirectory(small)
IF (ALBANY_LARGE_TESTS)
  add_subdirectory(large)
ENDIF()
