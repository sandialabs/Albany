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
ENDIF()

IF (ALBANY_MPI)
  set(Albany.exe          ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/Albany)
  set(AlbanyDakota.exe    ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanyDakota)
  set(AlbanySG.exe        ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanySG)
  set(AlbanyAnalysis.exe  ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
  set(Albany8.exe          ${MPIEX} ${MPINPF} 8 ${Albany_BINARY_DIR}/src/Albany)
ELSE()
  set(Albany.exe          ${Albany_BINARY_DIR}/src/Albany)
  set(AlbanyDakota.exe    ${Albany_BINARY_DIR}/src/AlbanyDakota)
  set(AlbanySG.exe        ${Albany_BINARY_DIR}/src/AlbanySG)
  set(AlbanyAnalysis.exe  ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
ENDIF()

# Only use 2 proc's for Coupled problems, since they system is 2x2
if (ALBANY_ASCR)
  IF (ALBANY_MPI)
    set(AlbanyAdjoint.exe   ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanyAdjoint)
    set(AlbanySGAdjoint.exe ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanySGAdjoint)
    set(AlbanyCoupled.exe   ${MPIEX} ${MPINPF} 2 ${Albany_BINARY_DIR}/src/AlbanyCoupled)
    set(AlbanySGCoupled.exe ${MPIEX} ${MPINPF} 2 ${Albany_BINARY_DIR}/src/AlbanySGCoupled)
  ELSE()
    set(AlbanyAdjoint.exe   ${Albany_BINARY_DIR}/src/AlbanyAdjoint)
    set(AlbanySGAdjoing.exe ${Albany_BINARY_DIR}/src/AlbanySGAdjoint)
    set(AlbanyCoupled.exe   ${Albany_BINARY_DIR}/src/AlbanyCoupled)
    set(AlbanySGCoupled.exe ${Albany_BINARY_DIR}/src/AlbanySGCoupled)
  ENDIF()
  set(SerialAlbanyCoupled.exe  ${Albany_BINARY_DIR}/src/AlbanyCoupled)
  set(SerialAlbanySGCoupled.exe  ${Albany_BINARY_DIR}/src/AlbanySGCoupled)
endif()

set(SerialAlbany.exe  ${Albany_BINARY_DIR}/src/Albany)
set(SerialAlbanySG.exe  ${Albany_BINARY_DIR}/src/AlbanySG)
set(SerialAlbanyDakota.exe  ${Albany_BINARY_DIR}/src/AlbanyDakota)

# Tests
IF(ALBANY_LCM)
  set(PartitionTest.exe  ${Albany_BINARY_DIR}/src/LCM/PartitionTest)
  set(MPS.exe  ${Albany_BINARY_DIR}/src/LCM/MaterialPointSimulator)
ENDIF()

# Heat Transfer Problems ###############
  add_subdirectory(SteadyHeat1D)
  add_subdirectory(SteadyHeat2D)
  add_subdirectory(SteadyHeat3D)
  add_subdirectory(SteadyHeat1DEB)
  add_subdirectory(SteadyHeat2DEB)
  add_subdirectory(SteadyHeat2DSS)
  add_subdirectory(ContinuationHeat1D)
  add_subdirectory(Heat1DPeriodic)
  add_subdirectory(Heat2DTriangles)
  add_subdirectory(Heat1DWithSource)
  add_subdirectory(HeatQuadTri)
  add_subdirectory(SGBratu1D)
  add_subdirectory(SGQuad2D)
  add_subdirectory(MPNIQuad2D)
  add_subdirectory(TransientHeat1D)
  add_subdirectory(TransientHeat2D)
  add_subdirectory(HeatEigenvalues)
  IF(ALBANY_SEACAS)
    add_subdirectory(Heat3DPamgen)
    add_subdirectory(Heat2DMMCylWithSource)
  #  add_subdirectory(TransientHeat2DTableSource)
    add_subdirectory(Ioss2D)
    add_subdirectory(Ioss3D)
    add_subdirectory(IossRestart)
    add_subdirectory(SteadyHeat2DInternalNeumann)  
    add_subdirectory(SteadyHeat2DRobin)  
  ENDIF()

  IF(ALBANY_ENABLE_CUBIT)
    add_subdirectory(CUTR_Heat1D)
    add_subdirectory(CUTR_Heat2D)
    add_subdirectory(CUTR_HeatStruct2D)
  ENDIF()

  IF(ALBANY_SCOREC)
    add_subdirectory(Heat2DFMDB)
  ENDIF()


# DEMO PDES ###############
IF(ALBANY_DEMO_PDES)
  add_subdirectory(Helmholtz2D)
  add_subdirectory(GPAM)
  add_subdirectory(LinComprNS)
  add_subdirectory(ComprNS)
  add_subdirectory(ODE)
  add_subdirectory(TekoPrec)

  # These require exodus, Ioss, stk_io, to read meshes
  IF(ALBANY_SEACAS)
    add_subdirectory(CahnHillElast2D)
    add_subdirectory(NSPoiseuille2D)
    add_subdirectory(NSRayleighBernard2D)
    add_subdirectory(NSRayleighBernard2DwithNeut)
    add_subdirectory(NSVortexShedding2D)  
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
  add_subdirectory(DomainTear2D)
  add_subdirectory(RandomFracture3D)
  add_subdirectory(ProjectionElasticity3D)
  #  add_subdirectory(GursonFD3D)
  add_subdirectory(CapModelPlasticity3D)
  add_subdirectory(StaticElasticity1D)
  add_subdirectory(StaticElasticity2D)
  add_subdirectory(StaticElasticity3D)
  add_subdirectory(Elasticity3DPressureBC)
  add_subdirectory(Elasticity2DTriangles)
  add_subdirectory(NonlinearElasticity3D)
  add_subdirectory(NonlinearElasticity2D)
  add_subdirectory(TransientElasticity2D)
  add_subdirectory(DiffusionDeformation3D)
  add_subdirectory(PoroElasticity2D)
  add_subdirectory(PoroElasticity3D)
  add_subdirectory(UnSatPoroElasticity3D)
  add_subdirectory(TLPoroElasticity2D)
  add_subdirectory(TLPoroElasticity3D)
  add_subdirectory(ThermoElasticity2D)
  add_subdirectory(ThermoElasticity3D)
  add_subdirectory(ThermoMechanical3D)
  add_subdirectory(ThermoPoroMechanics3D)
  add_subdirectory(J2Plasticity2D)	
  add_subdirectory(UnitTests)
  add_subdirectory(Partition)
  add_subdirectory(GradientDamage)
  add_subdirectory(TorsionBC)
  # add_subdirectory(J2Fiber)
  # add_subdirectory(RIHMR)

  IF(ALBANY_SEACAS)
    add_subdirectory(MaterialPointSimulator)
    add_subdirectory(BoreDemo)
    add_subdirectory(KfieldBC)
    add_subdirectory(HydrogenKfieldBC)
    add_subdirectory(Mechanics)
    add_subdirectory(MechanicsPorePressure)
#    add_subdirectory(MechanicsWithHydrogen)
    add_subdirectory(MechanicsRigidBody)
    add_subdirectory(SurfaceElement)
# Needs SEACAS binary tools - epu, exodiff
    add_subdirectory(Elasticity2DTraction)
  ENDIF()
  IF(ALBANY_LAME)
    add_subdirectory(LameStaticElasticity3D)
    add_subdirectory(LameMultiMaterials)
  ENDIF()
  IF(ALBANY_LAMENT)
    add_subdirectory(LamentStaticElasticity3D)
  ENDIF()

  IF(ALBANY_HYDRIDE)
    IF(ALBANY_MPI)
      add_subdirectory(MultiScaleRing)
    ENDIF()
  ENDIF()

ENDIF(ALBANY_LCM)

# QCAD ##################

IF(ALBANY_QCAD AND ALBANY_SEACAS)
  add_subdirectory(Poisson)
  add_subdirectory(Schrodinger)
  add_subdirectory(PoissonSchrodinger)
  add_subdirectory(PoissonSchroMosCap1D)
  add_subdirectory(ThermoElectrostatics2D)
ENDIF()

# FELIX ##################

IF(ALBANY_FELIX AND ALBANY_SEACAS )
  add_subdirectory(FELIX_Stokes)
ENDIF()

# MOR   ##################

IF(ALBANY_MOR)
  add_subdirectory(MOR)
ENDIF()

###################
