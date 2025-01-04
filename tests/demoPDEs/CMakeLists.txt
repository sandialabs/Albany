##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

  add_subdirectory(Helmholtz2D)
  add_subdirectory(AdvDiff)
  add_subdirectory(ReactDiffSystem)
  add_subdirectory(PoissonAdvDiffSystem)
  add_subdirectory(CoupledPoissonAdvDiffSystem)
  add_subdirectory(Poisson2D)
  add_subdirectory(ODE)
  add_subdirectory(ThermoElectrostatics2D)
  if(ALBANY_STK_EXPR_EVAL)
    add_subdirectory(Advection1D)
  endif()

  # These require exodus, Ioss, stk_io, to read meshes
  if(ALBANY_SEACAS)
    add_subdirectory(NSPoiseuille2D)
    if(NOT Kokkos_ENABLE_CUDA) # Currently takes too long to run on Ride
      add_subdirectory(NSVortexShedding2D)
    endif()
  endif()

  if(ALBANY_STK_EXPR_EVAL)
    add_subdirectory(Thermal1D)
    add_subdirectory(Thermal2D)
    add_subdirectory(Thermal3D)
  endif()

