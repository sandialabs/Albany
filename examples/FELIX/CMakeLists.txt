add_subdirectory(ExoMeshes)

add_subdirectory(AsciiMeshes)

add_subdirectory(FO_MMS)
add_subdirectory(FO_ISMIP)
add_subdirectory(FO_Test)
add_subdirectory(FO_GIS)
add_subdirectory(FO_Thermo)
add_subdirectory(Stokes_MMS)
add_subdirectory(Stokes_ISMIP)
add_subdirectory(Stokes_Test)
add_subdirectory(L1L2_MMS)

IF(ENABLE_CISM_INTERFACE)
add_subdirectory(CismAlbany)
ENDIF() 

add_subdirectory(Hydrology)

add_subdirectory(Enthalpy)
