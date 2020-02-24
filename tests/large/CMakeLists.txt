# DEMO PDES ###############
IF(ALBANY_DEMO_PDES)
  # These require exodus, Ioss, stk_io, to read meshes
  IF(ALBANY_SEACAS)
    add_subdirectory(NSVortexShedding)
  ENDIF()
ENDIF(ALBANY_DEMO_PDES)

# LCM ###############

IF(ALBANY_LCM)
  add_subdirectory(LCM)
ENDIF(ALBANY_LCM)

# TDM ###############

IF(ALBANY_TDM)
  add_subdirectory(TDM)
ENDIF(ALBANY_TDM)

# Aeras ##################

IF(ALBANY_AERAS)
  add_subdirectory(Aeras)
ENDIF()

# LandIce ##################

IF(ALBANY_LANDICE AND ALBANY_SEACAS)
  add_subdirectory(LandIce)
ENDIF()

# AMP   ##################

IF(ALBANY_AMP)
  add_subdirectory(AMP)
ENDIF()

IF(ALBANY_PERFORMANCE_TESTS)
  add_subdirectory(PerformanceTests)
ENDIF()
