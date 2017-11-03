 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_BE_nu8e15_vort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_BE_nu8e15_vort_T.xml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_BE_nu8e15_novort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_BE_nu8e15_novort_T.xml COPYONLY)               
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_RK4_nu8e15_novort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_RK4_nu8e15_novort_T.xml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitHV_BE_nu8e15_vort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitHV_BE_nu8e15_vort_T.xml COPYONLY)                                             

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitHV_BE_nu8e15_novort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitHV_BE_nu8e15_novort_T.xml COPYONLY)  
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_RK4_noAD_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_RK4_noAD_T.xml COPYONLY)  
               
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hv_T.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_hv_T.xml COPYONLY)               
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_hv_T.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_hv_T.xml COPYONLY)              
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitTimeInt.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitTimeInt.xml COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitTimeInt.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitTimeInt.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD9/uniform_16_quad9.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad9.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
#add_test(Aeras_${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
#add_test(Aeras_${testName}_HV_Tpetra ${AlbanyT.exe} input_hv_T.xml)


IF(NOT ALBANY_CHECK_FPE)
IF (NOT ALBANY_STRONG_FPE_CHECK) 
add_test(Aeras_${testName}_explicitHV_nu8e15_vort_BackwardEuler ${AlbanyT.exe} input_explicitHV_BE_nu8e15_vort_T.xml)
ENDIF()
ENDIF()
add_test(Aeras_${testName}_explicitHV_nu8e15_novort_BackwardEuler ${AlbanyT.exe} input_explicitHV_BE_nu8e15_novort_T.xml)

if (ALBANY_EPETRA)
add_test(Aeras_${testName}_explicitHV_nu8e15_novort_RungeKutta4 ${AlbanyT.exe} input_explicitHV_RK4_nu8e15_novort_T.xml)
endif() 

add_test(Aeras_${testName}_noAD_RungeKutta4 ${AlbanyT.exe} input_RK4_noAD_T.xml)

add_test(Aeras_${testName}_implicitHV_nu8e15_vort_BackwardEuler ${AlbanyT.exe} input_implicitHV_BE_nu8e15_vort_T.xml)
add_test(Aeras_${testName}_implicitHV_nu8e15_novort_BackwardEuler ${AlbanyT.exe} input_implicitHV_BE_nu8e15_novort_T.xml)

