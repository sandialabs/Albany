
# Copy files used by multiple tests
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)

# Get testRoot as QCAD_<the directory name>
get_filename_component(subRoot ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testRoot "QCAD_${subRoot}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_psci_mosdot_2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_psci_mosdot_2D.xml COPYONLY)

# Create tests with this name and standard executable
add_test(${testRoot}_mosdot_2D ${Albany.exe} input_psci_mosdot_2D.xml)

