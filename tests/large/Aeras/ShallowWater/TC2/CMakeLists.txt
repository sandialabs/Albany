 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSpectralT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSpectralT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_24elesSpectralT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_24elesSpectralT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSpectralRythmosSolver_RK4_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSpectralRythmosSolver_RK4_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSpectralRythmosSolver_KandG_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSpectralRythmosSolver_KandG_T.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(Aeras_${testName}_Spectral_Large_Quad9_BackwardEuler ${Albany.exe}
inputSpectralT.yaml) 
set_tests_properties(Aeras_${testName}_Spectral_Large_Quad9_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")
add_test(Aeras_${testName}_Spectral_24Eles_Quad25_BackwardEuler ${Albany.exe}
input_24elesSpectralT.yaml) 
set_tests_properties(Aeras_${testName}_Spectral_24Eles_Quad25_BackwardEuler PROPERTIES LABELS "Aeras;Tpetra;Forward")

#add_test(Aeras_${testName}_Spectral_RythmosSolver_RungeKutta4 ${Albany.exe}
#    inputSpectralRythmosSolverT.yaml) 
