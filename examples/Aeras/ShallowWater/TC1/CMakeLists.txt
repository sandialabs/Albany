 

# 1. Copy Input file from source to binary dir
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_slotcyl_hv_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_slotcyl_hv_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slotcyl_hv_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_slotcyl_hv_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/uniform_10.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_10.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
#if (ALBANY_EPETRA) 
#add_test(Aeras_${testName} ${Albany.exe})
#endif()
if (NOT ALBANY_EPETRA)
add_test(Aeras_${testName}_SlotCyl_HV_Tpetra ${AlbanyT.exe} input_slotcyl_hv_T.xml)
endif()
