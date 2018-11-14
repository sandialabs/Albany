configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_gis20km_refine.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_gis20km_refine.yaml COPYONLY)

#IK, 10/29/13: Convert to tpetra!
IF(ALBANY_STK_PERCEPT AND ALBANY_EPETRA)
add_test(${testName}_Gis20km_refine ${Albany8.exe} input_fo_gis20km_refine.yaml)
ENDIF()
