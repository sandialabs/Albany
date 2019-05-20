 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectralT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectralT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np3_RK4_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np3_RK4_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np3_BE_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np3_BE_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np4_BE_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np4_BE_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np5_BE_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np5_BE_T.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

add_test(Aeras_${testName}_Spectral_np2_RungeKutta4 ${Albany.exe} input_spectralT.yaml)
set_tests_properties(Aeras_${testName}_Spectral_np2_RungeKutta4 PROPERTIES LABELS "Aeras;Tpetra;Forward")
IF(NOT ALBANY_CHECK_FPE)
IF(NOT ALBANY_STRONG_FPE_CHECK) 
IF(NOT Kokkos_ENABLE_Cuda) 
add_test(Aeras_${testName}_Spectral_np4_BackwardEuler ${Albany8.exe} input_spectral_np4_BE_T.yaml) 
set_tests_properties(Aeras_${testName}_Spectral_np4_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")
ENDIF()
ENDIF()
ENDIF()
