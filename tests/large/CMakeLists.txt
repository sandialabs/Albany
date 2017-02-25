# DEMO PDES ###############
IF(ALBANY_DEMO_PDES)
  # These require exodus, Ioss, stk_io, to read meshes
  IF(ALBANY_SEACAS)
    add_subdirectory(NSVortexShedding)
  ENDIF()
ENDIF(ALBANY_DEMO_PDES)
