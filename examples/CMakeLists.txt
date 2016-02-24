##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Setup paths for SEACAS tools needed to run some of the tests
IF (ALBANY_SEACAS)
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

# Paths to the actual executables
set(AlbanyPath                         ${Albany_BINARY_DIR}/src/Albany)
set(AlbanyTPath                        ${Albany_BINARY_DIR}/src/AlbanyT)
set(AlbanySGPath                       ${Albany_BINARY_DIR}/src/AlbanySG)
set(AlbanyDakotaPath                   ${Albany_BINARY_DIR}/src/AlbanyDakota)
set(AlbanyDakotaTPath                  ${Albany_BINARY_DIR}/src/AlbanyDakotaT)
set(AlbanyAnalysisPath                 ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
set(AlbanyAnalysisTPath                ${Albany_BINARY_DIR}/src/AlbanyAnalysisT)
set(AlbanyAdjointPath                  ${Albany_BINARY_DIR}/src/AlbanyAdjoint)
set(AlbanySGAdjointPath                ${Albany_BINARY_DIR}/src/AlbanySGAdjoint)
set(AlbanyCoupledPath                  ${Albany_BINARY_DIR}/src/AlbanyCoupled)
set(AlbanySGCoupledPath                ${Albany_BINARY_DIR}/src/AlbanySGCoupled)

IF (ALBANY_MPI)
  set(SerialAlbany.exe                 ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanyPath})
  set(SerialAlbanyT.exe                ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanyTPath})
  set(SerialAlbanySG.exe               ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanySGPath})
  set(SerialAlbanyAnalysis.exe         ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanyAnalysisPath})
  set(SerialAlbanyDakota.exe           ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${MPIPOST} ${AlbanyDakotaPath})
  set(Albany.exe                       ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyPath})
  set(AlbanyT.exe                      ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyTPath})
  set(AlbanyDakota.exe                 ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyDakotaPath})
  set(AlbanyDakotaT.exe                ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyDakotaTPath})
  set(AlbanySG.exe                     ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanySGPath})
  set(AlbanyAnalysis.exe               ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyAnalysisPath})
  set(AlbanyAnalysisT.exe              ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${MPIPOST} ${AlbanyAnalysisTPath})
  set(Albany8.exe                      ${MPIEX} ${MPIPRE} ${MPINPF} 8 ${MPIPOST} ${AlbanyPath})
  set(AlbanyT8.exe                     ${MPIEX} ${MPIPRE} ${MPINPF} 8 ${MPIPOST} ${AlbanyTPath})
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

IF(ALBANY_HAVE_STK) # STK is needed for all these

# Heat Transfer Problems ###############
  add_subdirectory(SteadyHeat1D)
  add_subdirectory(SteadyHeat2D)
  add_subdirectory(SteadyHeat3D)
  add_subdirectory(SteadyHeat1DEB)
  add_subdirectory(SteadyHeat2DEB)
  add_subdirectory(ContinuationHeat1D)
  add_subdirectory(Heat1DPeriodic)
  add_subdirectory(Heat2DTriangles)
  add_subdirectory(Heat1DWithSource)
  add_subdirectory(SGBratu1D)
  add_subdirectory(SGQuad2D)
  add_subdirectory(MPNIQuad2D)
  add_subdirectory(TransientHeat1D)
  add_subdirectory(TransientHeat2D)
  add_subdirectory(HeatEigenvalues)
  IF(ALBANY_SEACAS)
    IF(ALBANY_PAMGEN)
      add_subdirectory(Heat3DPamgen)
    ENDIF()
    add_subdirectory(Heat2DMMCylWithSource)
    add_subdirectory(HeatQuadTri)
  #  add_subdirectory(TransientHeat2DTableSource)
    add_subdirectory(Ioss2D)
    add_subdirectory(Ioss3D)
    add_subdirectory(IossRestart)
    add_subdirectory(SteadyHeat2DInternalNeumann)
    add_subdirectory(SteadyHeat2DRobin)
    add_subdirectory(SteadyHeat2DSS)
  ENDIF(ALBANY_SEACAS)

  IF(ALBANY_ENABLE_CUBIT)
    add_subdirectory(CUTR_Heat1D)
    add_subdirectory(CUTR_Heat2D)
    add_subdirectory(CUTR_HeatStruct2D)
  ENDIF(ALBANY_ENABLE_CUBIT)
ENDIF(ALBANY_HAVE_STK)

IF(ALBANY_SCOREC)
  add_subdirectory(Heat3DPUMI)
ENDIF()

IF(ALBANY_HAVE_STK)
# DEMO PDES ###############
IF(ALBANY_DEMO_PDES)
  add_subdirectory(Helmholtz2D)
  add_subdirectory(LinComprNS)
  add_subdirectory(AdvDiff)
  add_subdirectory(ComprNS)
  add_subdirectory(ODE)
  add_subdirectory(TekoPrec)

  # These require exodus, Ioss, stk_io, to read meshes
  IF(ALBANY_SEACAS)
    add_subdirectory(GPAM)
    add_subdirectory(CahnHillElast2D)
    add_subdirectory(NSPoiseuille2D)
    add_subdirectory(NSRayleighBernard2D)
# Test appears to be unstable FIXME
#    add_subdirectory(NSRayleighBernard2DwithNeut)
    add_subdirectory(NSVortexShedding2D)
    add_subdirectory(PNP)
  ENDIF(ALBANY_SEACAS)

  if (ALBANY_ASCR)
    add_subdirectory(CoupledReactor)
    add_subdirectory(NSReactor)
  endif(ALBANY_ASCR)

  IF(ALBANY_ENABLE_CUBIT)
    add_subdirectory(HTE3D)
  ENDIF(ALBANY_ENABLE_CUBIT)

ENDIF(ALBANY_DEMO_PDES)

ENDIF(ALBANY_HAVE_STK)

# LCM ###############

IF(ALBANY_LCM)
  add_subdirectory(LCM)
ENDIF(ALBANY_LCM)

# GOAL ##############

IF(ALBANY_GOAL)
  add_subdirectory(GOAL)
ENDIF()

IF(ALBANY_HAVE_STK)

# QCAD ##################

IF(ALBANY_QCAD AND ALBANY_SEACAS)
  IF (ALBANY_EPETRA)
    add_subdirectory(QCAD)
  ENDIF()
  add_subdirectory(ThermoElectrostatics2D)
ENDIF()

# ATO ##################

IF(ALBANY_ATO)
  add_subdirectory(ATO)
ENDIF()

# FELIX ##################

IF(ALBANY_FELIX AND ALBANY_SEACAS )
  add_subdirectory(FELIX)
ENDIF()

# Aeras ##################

IF(ALBANY_AERAS)
  add_subdirectory(Aeras)
ENDIF()

# MOR   ##################

IF(ALBANY_MOR AND ALBANY_EPETRA)
  add_subdirectory(MOR)
ENDIF()

# AMP   ##################

IF(ALBANY_AMP)
  add_subdirectory(AMP)
ENDIF()

IF(ALBANY_PERFORMANCE_TESTS)
  add_subdirectory(PerformanceTests)
ENDIF()

ENDIF(ALBANY_HAVE_STK)

###################
