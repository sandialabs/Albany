##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

IF(ALBANY_STK) # STK is needed for all these

  # Heat Transfer Problems ###############
  add_subdirectory(SteadyHeat1D)
  add_subdirectory(SteadyHeat2D)
  add_subdirectory(SteadyHeatConstrainedOpt2D)
  add_subdirectory(SteadyHeat3D)
IF (ALBANY_PAMGEN) 
  add_subdirectory(SteadyHeat3DTest)
ENDIF ()
  add_subdirectory(SteadyHeat1DEB)
  add_subdirectory(SteadyHeat2DEB)
  add_subdirectory(ContinuationHeat1D)
  add_subdirectory(Heat1DPeriodic)
  add_subdirectory(Heat2DTriangles)
  add_subdirectory(Heat1DWithSource)
  add_subdirectory(MPNIQuad2D)
IF(ALBANY_EXPR_EVAL_SDBC)
  add_subdirectory(Thermal1D)
ENDIF()
  add_subdirectory(TransientHeat1D)
  add_subdirectory(TransientHeat2D)
  add_subdirectory(HeatEigenvalues)
  add_subdirectory(SideSetLaplacian) # Not 100% sure this requires STK, but I think so

  IF(ALBANY_SEACAS)
    IF(ALBANY_PAMGEN)
      add_subdirectory(Heat3DPamgen)
    ENDIF()
    add_subdirectory(Heat2DMMCylWithSource)
    add_subdirectory(HeatQuadTri)
    add_subdirectory(TransientHeat2DTableSource)
    add_subdirectory(Ioss2D)
    add_subdirectory(Ioss3D)
    add_subdirectory(IossRestart)
    add_subdirectory(SteadyHeat2DInternalNeumann)
    add_subdirectory(SteadyHeat2DRobin)
    add_subdirectory(SteadyHeat2DSS)
  ENDIF(ALBANY_SEACAS)

  add_subdirectory(Utils)
ENDIF(ALBANY_STK)

IF(ALBANY_SCOREC)
  add_subdirectory(Heat3DPUMI)
ENDIF()

IF(ALBANY_STK)
  # DEMO PDES ###############
  IF(ALBANY_DEMO_PDES)
    add_subdirectory(Helmholtz2D)
    add_subdirectory(LinComprNS)
    add_subdirectory(AdvDiff)
    add_subdirectory(ReactDiffSystem)
    add_subdirectory(ComprNS)
    add_subdirectory(ODE)
    add_subdirectory(ThermoElectrostatics2D)
    # add_subdirectory(TekoPrec)

    # These require exodus, Ioss, stk_io, to read meshes
    IF(ALBANY_SEACAS)
      add_subdirectory(CahnHillElast2D)
      add_subdirectory(NSPoiseuille2D)
      add_subdirectory(NSRayleighBernard2D)
      IF(NOT ALBANY_ENABLE_CUDA) # Currently takes too long to run on Ride
        add_subdirectory(NSVortexShedding2D)
      ENDIF()
      add_subdirectory(PNP)
    ENDIF()

    if (ALBANY_ASCR)
      add_subdirectory(CoupledReactor)
      add_subdirectory(NSReactor)
    endif()

  ENDIF(ALBANY_DEMO_PDES)

ENDIF(ALBANY_STK)

# LCM ###############
IF(ALBANY_LCM)
   add_subdirectory(LCM)
ENDIF()

IF(ALBANY_STK)

  # ATO ##################
  IF(ALBANY_ATO)
    add_subdirectory(ATO)
  ENDIF()

  # LandIce ##################
  IF(ALBANY_LANDICE AND ALBANY_SEACAS )
    add_subdirectory(LandIce)
  ENDIF()

  # LandIce ##################
  IF(ALBANY_TSUNAMI AND ALBANY_SEACAS )
    # add_subdirectory(Tsunami)
  ENDIF()

ENDIF()

###################
