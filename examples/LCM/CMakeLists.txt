##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

###################
set(PartitionTest.exe ${Albany_BINARY_DIR}/src/LCM/PartitionTest)
set(Subdivision.exe   ${Albany_BINARY_DIR}/src/LCM/Subdivision)
set(MPS.exe           ${Albany_BINARY_DIR}/src/LCM/MaterialPointSimulator)

IF(ALBANY_SEACAS)
  add_subdirectory(CohesiveElement)
  add_subdirectory(SurfaceElement)
  add_subdirectory(SurfaceElementLocking)
  add_subdirectory(StabilizedTet4)
  add_subdirectory(KfieldBC)
  add_subdirectory(HydrogenKfieldBC)
  add_subdirectory(TorsionBC)
  add_subdirectory(Mechanics)
  add_subdirectory(MechanicsPorePressure)
  add_subdirectory(MechanicsWithHydrogen)
  add_subdirectory(MechanicsWithTemperature)
  add_subdirectory(MechanicsRigidBody)
  add_subdirectory(Dynamics)
  add_subdirectory(Elasticity2DTraction)
  add_subdirectory(Partition)
  add_subdirectory(MaterialPointSimulator)
  add_subdirectory(Subdivision)
  add_subdirectory(BoreDemo)
ENDIF()

add_subdirectory(Catalyst)
add_subdirectory(DomainTear2D)
add_subdirectory(RandomFracture3D)
add_subdirectory(ProjectionElasticity3D)
add_subdirectory(CapModelPlasticity3D)
add_subdirectory(StaticElasticity1D)
add_subdirectory(StaticElasticity2D)
add_subdirectory(StaticElasticity3D)
add_subdirectory(Elasticity3DPressureBC)
add_subdirectory(Elasticity2DTriangles)
add_subdirectory(TransientElasticity2D)
add_subdirectory(PoroElasticity2D)
add_subdirectory(PoroElasticity3D)
add_subdirectory(UnSatPoroElasticity3D)
add_subdirectory(TLPoroElasticity2D)
add_subdirectory(TLPoroElasticity3D)
add_subdirectory(ThermoElasticity2D)
add_subdirectory(ThermoElasticity3D)
add_subdirectory(ThermoMechanical3D)
add_subdirectory(ThermoPoroMechanics3D)
add_subdirectory(UnitTests)
add_subdirectory(GradientDamage)

IF(ALBANY_LAME)
    add_subdirectory(LameStaticElasticity3D)
    add_subdirectory(LameMultiMaterials)
ENDIF()

IF(ALBANY_LAMENT)
  add_subdirectory(LamentStaticElasticity3D)
ENDIF()

IF(ALBANY_HYDRIDE)
  add_subdirectory(LaplaceBeltrami)
  IF(ALBANY_MPI)
    add_subdirectory(MultiScaleRing)
  ENDIF()
ENDIF()

IF(ALBANY_SEACAS OR ALBANY_SCOREC)
  add_subdirectory(MechanicsTensileTet10)
ENDIF()

IF(ALBANY_SCOREC)
  add_subdirectory(ElastMeshAdapt3D)
  add_subdirectory(J2PlasticityFMDBTet10)
  add_subdirectory(StaticElasticity3DFMDB)
ENDIF()

IF(ALBANY_STK_PERCEPT)
  add_subdirectory(Heat2DSTKAdapt)
  add_subdirectory(Necking3DSTKAdapt)
ENDIF()

###################
