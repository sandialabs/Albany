add_subdirectory(ExoMeshes)
add_subdirectory(AsciiMeshes)
add_subdirectory(FO_AIS)

IF (Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_CUDA_UVM)
  RETURN() # don't add any tests past this point if this is a CUDA build with uvm disabled
ENDIF()

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

# Some ad-hoc problems to quickly verify certain evaluators
add_subdirectory(unit)
