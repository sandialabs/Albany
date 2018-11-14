# 
# 1. Copy input file from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2d_ebw.yaml ${CMAKE_CURRENT_BINARY_DIR}/2d_ebw.yaml COPYONLY)
ENDIF() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2d_ebwT.yaml ${CMAKE_CURRENT_BINARY_DIR}/2d_ebwT.yaml COPYONLY)
# 2. Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2d_ebw.gen     ${CMAKE_CURRENT_BINARY_DIR}/2d_ebw.gen COPYONLY)
# 3. Copy runtest.cmake from source to binary dir

set (PARALLEL,  DEFINED MPIMNP AND ${MPIMNP} GREATER 1 )
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

IF (ALBANY_EPETRA)
IF (NOT ${PARALLEL}) 
IF (ATO_NLOPT)
add_test(ATO:${testName} ${Albany.exe} 2d_ebw.yaml)
ENDIF() 
ENDIF() 
ENDIF() 


IF (NOT ${PARALLEL}) 
IF (ATO_NLOPT)
add_test(ATOT:${testName} ${AlbanyT.exe} 2d_ebwT.yaml)
ENDIF() 
ENDIF() 
