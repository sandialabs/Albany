
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening.g
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DislocationDensityHardening.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/DislocationDensityHardening.gold.exo COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 SET(OUTFILE "DislocationDensityHardening.exo")
 SET(REF_FILE "DislocationDensityHardening.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=DislocationDensityHardening -DTEST_ARGS=DislocationDensityHardening.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})

endif()
