add_subdirectory(ExoMeshes)

add_subdirectory(AsciiMeshes)

add_subdirectory(FO_MMS)
add_subdirectory(FO_ISMIP)
add_subdirectory(FO_Test)
add_subdirectory(FO_GIS)
add_subdirectory(FO_AIS)
add_subdirectory(FO_Thermo)

add_subdirectory(SHMIP)

add_subdirectory(Enthalpy)

if (NOT ENABLE_MESH_DEPENDS_ON_PARAMETERS)
  add_subdirectory(FO_Hydrology)
endif()

IF(ENABLE_CISM_INTERFACE)
add_subdirectory(CismAlbany)
ENDIF() 
