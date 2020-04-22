# DEMO PDES ###############
IF(ALBANY_DEMO_PDES)
  # These require exodus, Ioss, stk_io, to read meshes
  IF(ALBANY_SEACAS)
    add_subdirectory(NSVortexShedding)
  ENDIF()
ENDIF(ALBANY_DEMO_PDES)

# LandIce ##################

IF(ALBANY_LANDICE AND ALBANY_SEACAS)
  add_subdirectory(LandIce)
ENDIF()

IF(ALBANY_PERFORMANCE_TESTS)
  add_subdirectory(PerformanceTests)
ENDIF()
