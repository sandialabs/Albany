
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily.g
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiFamily.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MultiFamily.gold.exo COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 SET(OUTFILE "MultiFamily.exo")
 SET(REF_FILE "MultiFamily.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=MultiFamily -DTEST_ARGS=MultiFamily.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
