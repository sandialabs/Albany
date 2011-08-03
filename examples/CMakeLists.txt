#/********************************************************************\
#*            Albany, Copyright (2010) Sandia Corporation             *
#*                                                                    *
#* Notice: This computer software was prepared by Sandia Corporation, *
#* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
#* the Department of Energy (DOE). All rights in the computer software*
#* are reserved by DOE on behalf of the United States Government and  *
#* the Contractor as provided in the Contract. You are authorized to  *
#* use this computer software for Governmental purposes but it is not *
#* to be released or distributed to the public. NEITHER THE GOVERNMENT*
#* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
#* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
#* including this sentence must appear on any copies of this software.*
#*    Questions to Andy Salinger, agsalin@sandia.gov                  *
#\********************************************************************/

# 

IF (ALBANY_MPI)
  set(Albany.exe          ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/Albany)
  set(AlbanyDakota.exe    ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanyDakota)
  set(AlbanySG.exe        ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanySG)
  set(AlbanyAnalysis.exe  ${MPIEX} ${MPINPF} ${MPIMNP} ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
ELSE()
  set(Albany.exe          ${Albany_BINARY_DIR}/src/Albany)
  set(AlbanyDakota.exe    ${Albany_BINARY_DIR}/src/AlbanyDakota)
  set(AlbanySG.exe        ${Albany_BINARY_DIR}/src/AlbanySG)
  set(AlbanyAnalysis.exe  ${Albany_BINARY_DIR}/src/AlbanyAnalysis)
ENDIF()

set(SerialAlbany.exe  ${Albany_BINARY_DIR}/src/Albany)
set(SerialAlbanyDakota.exe  ${Albany_BINARY_DIR}/src/AlbanyDakota)

# Tests
IF(ALBANY_LCM)
  set(UtilsTest.exe  ${Albany_BINARY_DIR}/src/UtilsTest)
  set(PartitionTest.exe  ${Albany_BINARY_DIR}/src/PartitionTest)
ENDIF()

add_subdirectory(SteadyHeat1D)
add_subdirectory(SteadyHeat2D)
add_subdirectory(SteadyHeat3D)
add_subdirectory(ContinuationHeat1D)
add_subdirectory(Heat1DPeriodic)
add_subdirectory(Heat2DTriangles)
add_subdirectory(Heat3DPamgen)
add_subdirectory(Heat1DWithSource)
add_subdirectory(Helmholtz2D)

add_subdirectory(ODE)
add_subdirectory(Poisson)
add_subdirectory(ThermoElectrostatics2D)
add_subdirectory(HeatEigenvalues)
add_subdirectory(Schrodinger)

IF(ALBANY_LCM)
  add_subdirectory(StaticElasticity1D)
  add_subdirectory(StaticElasticity2D)
  add_subdirectory(StaticElasticity3D)
  add_subdirectory(Elasticity2DTriangles)
  add_subdirectory(NonlinearElasticity3D)
  add_subdirectory(NonlinearElasticity2D)
  add_subdirectory(TransientElasticity2D)
  add_subdirectory(ThermoElasticity2D)
  add_subdirectory(ThermoElasticity3D)
  add_subdirectory(J2Plasticity2D)	
  add_subdirectory(UtilsTest)
  add_subdirectory(Partition)
  add_subdirectory(GradientDamage)
  IF(ALBANY_LAME)
    add_subdirectory(LameStaticElasticity3D)
  ENDIF()
ENDIF()

IF(ALBANY_SEACAS)
  add_subdirectory(Ioss2D)
  add_subdirectory(Ioss3D)
  add_subdirectory(IossRestart)
  add_subdirectory(NSPoiseuille2D)
  add_subdirectory(NSRayleighBernard2D)
  
ENDIF()

IF(ALBANY_SEACAS AND ALBANY_LCM)
  add_subdirectory(BoreDemo)
  add_subdirectory(KfieldBC)
ENDIF()

add_subdirectory(SGBratu1D)
add_subdirectory(SGQuad2D)
add_subdirectory(MPNIQuad2D)
add_subdirectory(TransientHeat1D)
add_subdirectory(TransientHeat2D)
add_subdirectory(TekoPrec)

IF(ALBANY_ENABLE_CUBIT)
  add_subdirectory(CUTR_Heat1D)
  add_subdirectory(CUTR_Heat2D)
  add_subdirectory(CUTR_HeatStruct2D)
  add_subdirectory(HTE3D)
ENDIF()
