##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Heat Transfer Problems ###############
add_subdirectory(SteadyHeat2D)
add_subdirectory(SteadyHeatConstrainedOpt2D)
add_subdirectory(SteadyHeat3D)
add_subdirectory(SteadyHeat2DEB)
add_subdirectory(ContinuationHeat1D)
add_subdirectory(TransientHeat2D)
add_subdirectory(SideSetLaplacian) # Not 100% sure this requires STK, but I think so

IF(ALBANY_SEACAS)
  add_subdirectory(HeatQuadTri)
  add_subdirectory(Ioss3D)
  add_subdirectory(IossRestart)
  if (NOT ALBANY_DISABLE_OUTPUT_MESH)
    add_subdirectory(SteadyHeat2DRobin)
  endif()
  add_subdirectory(SteadyHeat2DSS)
ENDIF(ALBANY_SEACAS)

add_subdirectory(Utils)

