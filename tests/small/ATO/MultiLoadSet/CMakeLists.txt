# 
# 1. Copy input file from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_oc.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_oc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_nlopt.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_nlopt.yaml COPYONLY)
ENDIF() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_ocT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_ocT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_nloptT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_nloptT.yaml COPYONLY)
# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen     ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen.4.3 COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF (ALBANY_EPETRA) 
# 3. Copy runtest.cmake from source to binary dir
add_test(ATO:${testName}_Nodal_OC ${Albany.exe} nodal_oc.yaml)
IF (ATO_NLOPT) 
add_test(ATO:${testName}_Nodal_NLOPT ${Albany.exe} nodal_nlopt.yaml)
ENDIF() 
ENDIF() 
add_test(ATOT:${testName}_Nodal_OC ${AlbanyT.exe} nodal_ocT.yaml)
IF (ATO_NLOPT) 
add_test(ATOT:${testName}_Nodal_NLOPT ${AlbanyT.exe} nodal_nloptT.yaml)
ENDIF() 
