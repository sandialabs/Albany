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

# Aeras ##################

IF(ALBANY_AERAS)
  add_subdirectory(Aeras)
ENDIF()

# FELIX ##################

IF(ALBANY_FELIX AND ALBANY_SEACAS)
  add_subdirectory(FELIX)
ENDIF()

# AMP   ##################

IF(ALBANY_AMP)
  add_subdirectory(AMP)
ENDIF()

IF(ALBANY_PERFORMANCE_TESTS)
  add_subdirectory(PerformanceTests)
ENDIF()
