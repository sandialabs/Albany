 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_BE_nu8e15_vort_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_BE_nu8e15_vort_T.yaml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_BE_nu8e15_novort_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_BE_nu8e15_novort_T.yaml COPYONLY)               
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_RK4_nu8e15_novort_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_RK4_nu8e15_novort_T.yaml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitHV_BE_nu8e15_vort_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitHV_BE_nu8e15_vort_T.yaml COPYONLY)                                             

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitHV_BE_nu8e15_novort_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitHV_BE_nu8e15_novort_T.yaml COPYONLY)  
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_RK4_noAD_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_RK4_noAD_T.yaml COPYONLY)  
               
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hv_T.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_hv_T.yaml COPYONLY)               
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_hv_T.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_hv_T.yaml COPYONLY)              
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitTimeInt.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitTimeInt.yaml COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitTimeInt.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitTimeInt.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD9/uniform_16_quad9.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad9.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
#add_test(Aeras_${testName}_Tpetra ${Albany.exe} inputT.yaml)
#add_test(Aeras_${testName}_HV_Tpetra ${Albany.exe} input_hv_T.yaml)


IF(NOT ALBANY_CHECK_FPE)
IF (NOT ALBANY_STRONG_FPE_CHECK) 
add_test(Aeras_${testName}_explicitHV_nu8e15_vort_BackwardEuler ${Albany.exe} input_explicitHV_BE_nu8e15_vort_T.yaml)
set_tests_properties(Aeras_${testName}_explicitHV_nu8e15_vort_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")
ENDIF()
ENDIF()
add_test(Aeras_${testName}_explicitHV_nu8e15_novort_BackwardEuler ${Albany.exe} input_explicitHV_BE_nu8e15_novort_T.yaml)
set_tests_properties(Aeras_${testName}_explicitHV_nu8e15_novort_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")

add_test(Aeras_${testName}_explicitHV_nu8e15_novort_RungeKutta4 ${Albany.exe} input_explicitHV_RK4_nu8e15_novort_T.yaml)
set_tests_properties(Aeras_${testName}_explicitHV_nu8e15_novort_RungeKutta4 PROPERTIES LABELS "Aeras;Tpetra;Forward")

add_test(Aeras_${testName}_noAD_RungeKutta4 ${Albany.exe} input_RK4_noAD_T.yaml)
set_tests_properties(Aeras_${testName}_noAD_RungeKutta4 PROPERTIES LABELS "Aeras;Tpetra;Forward")

add_test(Aeras_${testName}_implicitHV_nu8e15_vort_BackwardEuler ${Albany.exe} input_implicitHV_BE_nu8e15_vort_T.yaml)
set_tests_properties(Aeras_${testName}_implicitHV_nu8e15_vort_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")
#add_test(Aeras_${testName}_implicitHV_nu8e15_novort_BackwardEuler ${Albany.exe} input_implicitHV_BE_nu8e15_novort_T.yaml)
#set_tests_properties(Aeras_${testName}_implicitHV_nu8e15_novort_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")

