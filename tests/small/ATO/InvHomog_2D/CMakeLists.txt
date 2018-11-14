# 
# 1. Copy input file from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal.yaml COPYONLY)
ENDIF() 
# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RVE_restart.gen     ${CMAKE_CURRENT_BINARY_DIR}/RVE_restart.gen COPYONLY)
# 3. Copy runtest.cmake from source to binary dir
set (PARALLEL,  DEFINED MPIMNP AND ${MPIMNP} GREATER 1 )
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF (ALBANY_EPETRA)
IF (NOT ${PARALLEL}) 
IF (ATO_NLOPT)
add_test(ATO:${testName} ${Albany.exe} nodal.yaml)
ENDIF() 
ENDIF() 
ENDIF() 
