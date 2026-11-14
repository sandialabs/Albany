add_subdirectory(ExoMeshes)
add_subdirectory(AsciiMeshes)
if (ALBANY_OMEGAH)
  add_subdirectory(OmegahMeshes)
endif()
add_subdirectory(EllipticMockModel)
add_subdirectory(FO_AIS)
add_subdirectory(FO_MMS)
add_subdirectory(FO_ISMIP)
add_subdirectory(FO_Test)
add_subdirectory(FO_GIS)
add_subdirectory(FO_Thermo)

add_subdirectory(SHMIP)

add_subdirectory(Enthalpy)

if (ENABLE_EXPERIMENTAL AND NOT ENABLE_MESH_DEPENDS_ON_PARAMETERS)
  add_subdirectory(FO_Hydrology)
endif()

IF(ENABLE_CISM_INTERFACE)
  if (NOT CISM_EXE_DIR)
    message("\n CISM_EXE_DIR not defined.  CismAlbany test will not run.\n")
  else()
    message("  -- CISM_EXE_DIR ${CISM_EXE_DIR}")

    set(CismAlbanyPath  ${CISM_EXE_DIR}/cism_driver)
    add_subdirectory(CismAlbany)
  endif()
ENDIF()

# Unit and regression tests of the MPAS-Albany interface, called as MALI does through a mock of MPAS
IF(ENABLE_MPAS_INTERFACE)
  add_subdirectory(MPASInterface)
ENDIF()

# Some ad-hoc problems to quickly verify certain evaluators
add_subdirectory(unit)
