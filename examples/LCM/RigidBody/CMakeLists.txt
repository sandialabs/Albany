
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_exodiff_test.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_exodiff_test.py COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation.g
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation.xml
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation_Material.xml
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation_Material.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation.gold.exo COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(RigidBody_RigidRotation python run_exodiff_test.py "${SerialAlbanyT.exe}" RigidRotation.xml)
endif()
