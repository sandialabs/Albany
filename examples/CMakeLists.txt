##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
set(AlbanySGPath                       ${Albany_BINARY_DIR}/src/AlbanySG)
set(AlbanyDakotaPath                   ${Albany_BINARY_DIR}/src/AlbanyDakota)
set(AlbanyAnalysisPath                 ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
set(AlbanyAdjointPath                  ${Albany_BINARY_DIR}/src/AlbanyAdjoint)
set(AlbanySGAdjointPath                ${Albany_BINARY_DIR}/src/AlbanySGAdjoint)
set(AlbanyCoupledPath                  ${Albany_BINARY_DIR}/src/AlbanyCoupled)
set(AlbanySGCoupledPath                ${Albany_BINARY_DIR}/src/AlbanySGCoupled)

IF (ALBANY_MPI)
  set(SerialAlbany.exe                 ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${AlbanyPath})
  set(SerialAlbanySG.exe               ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${AlbanySGPath})
  set(SerialAlbanyDakota.exe           ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${AlbanyDakotaPath})
  set(Albany.exe                       ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${AlbanyPath})
  set(AlbanyDakota.exe                 ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${AlbanyDakotaPath})
  set(AlbanySG.exe                     ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${AlbanySGPath})
  set(AlbanyAnalysis.exe               ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${AlbanyAnalysisPath})
  set(Albany8.exe                      ${MPIEX} ${MPIPRE} ${MPINPF} 8 ${AlbanyPath})
ELSE()
  set(SerialAlbany.exe                 ${AlbanyPath})
  set(SerialAlbanySG.exe               ${AlbanySGPath})
  set(SerialAlbanyDakota.exe           ${AlbanyDakotaPath})
  set(Albany.exe                       ${AlbanyPath})
  set(AlbanyDakota.exe                 ${AlbanyDakotaPath})
  set(AlbanySG.exe                     ${AlbanySGPath})
  set(AlbanyAnalysis.exe               ${AlbanyAnalysisPath})
ENDIF()

# Only use 2 proc's for Coupled problems, since they system is 2x2
if (ALBANY_ASCR)
  IF (ALBANY_MPI)
    set(AlbanyAdjoint.exe   ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${AlbanyAdjointPath})
    set(AlbanySGAdjoint.exe ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${AlbanySGAdjointPath})
    set(AlbanyCoupled.exe   ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${AlbanyCoupledPath})
    set(AlbanySGCoupled.exe ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${AlbanySGCoupledPath})
  ELSE()
    set(AlbanyAdjoint.exe   ${AlbanyAdjointPath})
    set(AlbanySGAdjoing.exe ${AlbanySGAdjointPath})
    set(AlbanyCoupled.exe   ${AlbanyCoupledPath})
    set(AlbanySGCoupled.exe ${AlbanySGCoupledPath})
  ENDIF()
  set(SerialAlbanyCoupled.exe  ${AlbanyCoupledPath})
  set(SerialAlbanySGCoupled.exe  ${AlbanySGCoupledPath})
endif()


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
    add_subdirectory(Heat3DPamgen)
    add_subdirectory(Heat2DMMCylWithSource)
    add_subdirectory(HeatQuadTri)
  #  add_subdirectory(TransientHeat2DTableSource)
    add_subdirectory(Ioss2D)
    add_subdirectory(Ioss3D)
    add_subdirectory(IossRestart)
    add_subdirectory(SteadyHeat2DInternalNeumann)
    add_subdirectory(SteadyHeat2DRobin)
    add_subdirectory(SteadyHeat2DSS)
  ENDIF()

  IF(ALBANY_ENABLE_CUBIT)
    add_subdirectory(CUTR_Heat1D)
    add_subdirectory(CUTR_Heat2D)
    add_subdirectory(CUTR_HeatStruct2D)
  ENDIF()

  IF(ALBANY_SCOREC)
    add_subdirectory(Heat3DFMDB)
  ENDIF()


# DEMO PDES ###############
IF(ALBANY_DEMO_PDES)
  add_subdirectory(Helmholtz2D)
  add_subdirectory(LinComprNS)
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
  ENDIF()

  if (ALBANY_ASCR)
    add_subdirectory(CoupledReactor)
    add_subdirectory(NSReactor)
  endif()

  IF(ALBANY_ENABLE_CUBIT)
    add_subdirectory(HTE3D)
  ENDIF()

ENDIF(ALBANY_DEMO_PDES)

# LCM ###############
IF(ALBANY_LCM)
  add_subdirectory(LCM)
ENDIF(ALBANY_LCM)

# QCAD ##################

IF(ALBANY_QCAD AND ALBANY_SEACAS)
  add_subdirectory(QCAD)
  add_subdirectory(ThermoElectrostatics2D)
ENDIF()

# FELIX ##################

IF(ALBANY_FELIX AND ALBANY_SEACAS )
  add_subdirectory(FELIX_Stokes)
ENDIF()

# Aeras ##################

IF(ALBANY_AERAS)
  add_subdirectory(Aeras)
ENDIF()

# MOR   ##################

IF(ALBANY_MOR)
  add_subdirectory(MOR)
ENDIF()

IF(ALBANY_PERFORMANCE_TESTS)
  add_subdirectory(PerformanceTests)
ENDIF()

###################
