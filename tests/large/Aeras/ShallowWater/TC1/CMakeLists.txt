 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_homme_comp_BDF2_np4_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_homme_comp_BDF2_np4_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_homme_comp_BDF2_np7_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_homme_comp_BDF2_np7_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_homme_comp_RK3TVD_np4_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_homme_comp_RK3TVD_np4_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_homme_comp_RK3TVD_np7_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_homme_comp_RK3TVD_np7_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_homme_comp_RK3TVD_np4_hv_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_homme_comp_RK3TVD_np4_hv_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_homme_comp_RK3TVD_np7_hv_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_homme_comp_RK3TVD_np7_hv_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_slotcyl_hv_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_slotcyl_hv_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_slotcyl_hv_T_rk4.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_slotcyl_hv_T_rk4.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/uniform_10.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_10.g COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slotcyl_explHV_nu0_BEuler_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_slotcyl_explHV_nu0_BEuler_T.yaml COPYONLY)  
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slotcyl_explHV_nu1e17_BEuler_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_slotcyl_explHV_nu1e17_BEuler_T.yaml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slotcyl_explHV_nu0_RK4_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_slotcyl_explHV_nu0_RK4_T.yaml COPYONLY)    

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slotcyl_explHV_nu1e17_RK4_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_slotcyl_explHV_nu1e17_RK4_T.yaml COPYONLY)  
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slotcyl_implHV_nu0_BEuler_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_slotcyl_implHV_nu0_BEuler_T.yaml COPYONLY) 
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slotcyl_implHV_nu1e17_BEuler_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_slotcyl_implHV_nu1e17_BEuler_T.yaml COPYONLY)                

                            
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
#add_test(Aeras_${testName}_SlotCyl_HV_Tpetra ${Albany.exe} input_slotcyl_hv_T.yaml)

add_test(Aeras_${testName}_SlotCyl_implicitHV_nu0_BackwardEuler ${Albany.exe} input_slotcyl_implHV_nu0_BEuler_T.yaml)
set_tests_properties(Aeras_${testName}_SlotCyl_implicitHV_nu0_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")
add_test(Aeras_${testName}_SlotCyl_implicitHV_nu1e17_BackwardEuler ${Albany.exe} input_slotcyl_implHV_nu1e17_BEuler_T.yaml)
set_tests_properties(Aeras_${testName}_SlotCyl_implicitHV_nu1e17_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")

add_test(Aeras_${testName}_SlotCyl_explicitHV_nu0_BackwardEuler ${Albany.exe} input_slotcyl_explHV_nu0_BEuler_T.yaml)
set_tests_properties(Aeras_${testName}_SlotCyl_explicitHV_nu0_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")
add_test(Aeras_${testName}_SlotCyl_explicitHV_nu1e17_BackwardEuler ${Albany.exe} input_slotcyl_explHV_nu1e17_BEuler_T.yaml)
set_tests_properties(Aeras_${testName}_SlotCyl_explicitHV_nu1e17_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")

add_test(Aeras_${testName}_SlotCyl_explicitHV_nu0_RungeKutta4 ${Albany.exe} input_slotcyl_explHV_nu0_RK4_T.yaml)
set_tests_properties(Aeras_${testName}_SlotCyl_explicitHV_nu0_RungeKutta4 PROPERTIES LABELS "Aeras;Tpetra;Forward")

