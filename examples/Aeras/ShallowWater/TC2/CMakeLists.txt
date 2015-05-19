 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSpectralT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputSpectralT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_24elesT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_24elesT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_24elesSpectralT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_24elesSpectralT.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(Aeras_${testName} ${Albany.exe})
endif()
if (NOT ALBANY_EPETRA) 
add_test(Aeras_${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
add_test(Aeras_${testName}_Spectral_Large_Quad9_Tpetra ${AlbanyT.exe}
inputSpectralT.xml) 
add_test(Aeras_${testName}_24Eles_Quad9_Tpetra ${AlbanyT.exe}
input_24elesT.xml) 
add_test(Aeras_${testName}_Spectral_24Eles_Quad25_Tpetra ${AlbanyT.exe}
input_24elesSpectralT.xml) 
endif()
