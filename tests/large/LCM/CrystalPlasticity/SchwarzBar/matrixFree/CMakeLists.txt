
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
	        ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupling.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/coupling.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupling.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/coupling.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/coupling.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/coupling.gold.exo COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
 SET(OUTFILE "coupling.exo")
 SET(REF_FILE "coupling.gold.exo")
 add_test(NAME CrystalPlasticity_SchwarzBar_${testName}_np1
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=coupling -DTEST_ARGS=coupling.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
