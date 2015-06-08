
# Directories that fully depend on Exodus
IF(ALBANY_SEACAS)
add_subdirectory(ExoMeshes)
add_subdirectory(FO_GIS)
add_subdirectory(Stokes_Test)
ENDIF()

#These have some tests that depend on Exodus
add_subdirectory(FO_MMS)
add_subdirectory(FO_Test)
add_subdirectory(AsciiMeshes)

#These do not depend on Exodus
add_subdirectory(FO_ISMIP)
add_subdirectory(Stokes_MMS)
add_subdirectory(Stokes_ISMIP)
add_subdirectory(L1L2_MMS)
