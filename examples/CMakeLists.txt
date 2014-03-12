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

IF (ALBANY_MPI)
  set(SerialAlbany.exe    ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${Albany_BINARY_DIR}/src/Albany)
  set(SerialAlbanySG.exe    ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${Albany_BINARY_DIR}/src/AlbanySG)
  set(SerialAlbanyDakota.exe    ${MPIEX} ${MPIPRE} ${MPINPF} 1 ${Albany_BINARY_DIR}/src/AlbanyDakota)
  set(Albany.exe          ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/Albany)
  set(AlbanyDakota.exe    ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanyDakota)
  set(AlbanySG.exe        ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanySG)
  set(AlbanyAnalysis.exe  ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
  set(Albany8.exe          ${MPIEX} ${MPIPRE} ${MPINPF} 8 ${Albany_BINARY_DIR}/src/Albany)
ELSE()
  set(SerialAlbany.exe  ${Albany_BINARY_DIR}/src/Albany)
  set(SerialAlbanySG.exe  ${Albany_BINARY_DIR}/src/AlbanySG)
  set(SerialAlbanyDakota.exe  ${Albany_BINARY_DIR}/src/AlbanyDakota)
  set(Albany.exe          ${Albany_BINARY_DIR}/src/Albany)
  set(AlbanyDakota.exe    ${Albany_BINARY_DIR}/src/AlbanyDakota)
  set(AlbanySG.exe        ${Albany_BINARY_DIR}/src/AlbanySG)
  set(AlbanyAnalysis.exe  ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
ENDIF()

# Only use 2 proc's for Coupled problems, since they system is 2x2
if (ALBANY_ASCR)
  IF (ALBANY_MPI)
    set(AlbanyAdjoint.exe   ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanyAdjoint)
    set(AlbanySGAdjoint.exe ${MPIEX} ${MPIPRE} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanySGAdjoint)
    set(AlbanyCoupled.exe   ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${Albany_BINARY_DIR}/src/AlbanyCoupled)
    set(AlbanySGCoupled.exe ${MPIEX} ${MPIPRE} ${MPINPF} 2 ${Albany_BINARY_DIR}/src/AlbanySGCoupled)
  ELSE()
    set(AlbanyAdjoint.exe   ${Albany_BINARY_DIR}/src/AlbanyAdjoint)
    set(AlbanySGAdjoing.exe ${Albany_BINARY_DIR}/src/AlbanySGAdjoint)
    set(AlbanyCoupled.exe   ${Albany_BINARY_DIR}/src/AlbanyCoupled)
    set(AlbanySGCoupled.exe ${Albany_BINARY_DIR}/src/AlbanySGCoupled)
  ENDIF()
  set(SerialAlbanyCoupled.exe  ${Albany_BINARY_DIR}/src/AlbanyCoupled)
  set(SerialAlbanySGCoupled.exe  ${Albany_BINARY_DIR}/src/AlbanySGCoupled)
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
