 

#IK, 1/27/15
#This is for a toy problem on a 24 element mesh I am checking in to debug 
#the enrichment capabilities in the Aeras::Discretization class.

#Note that the enrichment can only be run right now when ALBANY_EPETRA_EXE is
#OFF.

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_shell9T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_shell9T.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
#if (ALBANY_EPETRA) 
#add_test(Aeras_${testName} ${Albany.exe})
#endif()
#add_test(Aeras_${testName}_Tpetra ${AlbanyT.exe} inputT.xml)

