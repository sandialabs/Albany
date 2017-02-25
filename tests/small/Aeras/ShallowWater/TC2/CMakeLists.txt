 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSpectralT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSpectralT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_24elesSpectralT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_24elesSpectralT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSpectralRythmosSolver_RK4_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSpectralRythmosSolver_RK4_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSpectralRythmosSolver_KandG_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSpectralRythmosSolver_KandG_T.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
add_test(Aeras_${testName}_Spectral_Large_Quad9_BackwardEuler ${AlbanyT.exe}
inputSpectralT.xml) 
add_test(Aeras_${testName}_Spectral_24Eles_Quad25_BackwardEuler ${AlbanyT.exe}
input_24elesSpectralT.xml) 

#add_test(Aeras_${testName}_Spectral_RythmosSolver_RungeKutta4 ${AlbanyT.exe}
#    inputSpectralRythmosSolverT.xml) 
