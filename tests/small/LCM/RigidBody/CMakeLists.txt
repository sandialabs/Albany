# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation.g
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RigidRotation.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/RigidRotation.gold.exo COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 SET(OUTFILE "RigidRotation.exo")
 SET(REF_FILE "RigidRotation.gold.exo")
 add_test(NAME ${testName}
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=RigidRotation -DTEST_ARGS=RigidRotation.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P runtest.cmake)

endif()
