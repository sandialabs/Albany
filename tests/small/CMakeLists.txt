##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Heat Transfer Problems ###############
add_subdirectory(SteadyHeat2D)
add_subdirectory(SteadyHeatConstrainedOpt2D)
add_subdirectory(SteadyHeat3D)
# add_subdirectory(SteadyHeat2DEB)
add_subdirectory(ContinuationHeat1D)
add_subdirectory(TransientHeat2D)
add_subdirectory(SideSetLaplacian) # Not 100% sure this requires STK, but I think so

IF(ALBANY_SEACAS)
  # add_subdirectory(HeatQuadTri)
  add_subdirectory(Ioss3D)
  add_subdirectory(IossRestart)
  # add_subdirectory(SteadyHeat2DRobin)
  add_subdirectory(SteadyHeat2DSS)
ENDIF(ALBANY_SEACAS)

add_subdirectory(Utils)

# DEMO PDES ###############
IF(ALBANY_DEMO_PDES)
  add_subdirectory(AdvDiff)
  add_subdirectory(ReactDiffSystem)
  add_subdirectory(ODE)
  add_subdirectory(ThermoElectrostatics2D)
  IF(ALBANY_STK_EXPR_EVAL)
    add_subdirectory(Thermal1D)
    add_subdirectory(Thermal2D)
    add_subdirectory(Thermal3D)
    add_subdirectory(Advection1D)
  ENDIF()

  # These require exodus, Ioss, stk_io, to read meshes
  IF(ALBANY_SEACAS)
    add_subdirectory(NSPoiseuille2D)
    IF(NOT ALBANY_ENABLE_CUDA) # Currently takes too long to run on Ride
      add_subdirectory(NSVortexShedding2D)
    ENDIF()
  ENDIF()

ENDIF(ALBANY_DEMO_PDES)

# LandIce ##################
IF(ALBANY_LANDICE AND ALBANY_SEACAS )
  add_subdirectory(LandIce)
ENDIF()

###################
