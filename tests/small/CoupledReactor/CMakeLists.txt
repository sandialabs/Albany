

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_heat.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_heat.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_neutronics.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_neutronics.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_domain_coupled.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_domain_coupled.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_network_coupled.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_network_coupled.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testNameDomain ${CMAKE_CURRENT_SOURCE_DIR}_Domain NAME)
get_filename_component(testNameNetwork ${CMAKE_CURRENT_SOURCE_DIR}_Network NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testNameDomain} ${Albany.exe} input_domain_coupled.xml)
add_test(${testNameNetwork} ${AlbanyCoupled.exe} input_network_coupled.xml)
endif()

