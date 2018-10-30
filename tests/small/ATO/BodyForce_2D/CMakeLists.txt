# 
# 1. Copy input file from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchell.yaml COPYONLY)
ENDIF() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchellT.yaml ${CMAKE_CURRENT_BINARY_DIR}/mitchellT.yaml COPYONLY)
# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell.gen     ${CMAKE_CURRENT_BINARY_DIR}/mitchell.gen COPYONLY)


set (PARALLEL,  DEFINED MPIMNP AND ${MPIMNP} GREATER 1 )
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF (ALBANY_EPETRA)
IF (NOT ${PARALLEL}) 
IF (ATO_NLOPT)
add_test(ATO:${testName} ${Albany.exe} mitchell.yaml)
ENDIF() 
ENDIF() 
ENDIF() 


IF (NOT ${PARALLEL}) 
IF (ATO_NLOPT)
add_test(ATOT:${testName} ${AlbanyT.exe} mitchellT.yaml)
ENDIF() 
ENDIF() 


