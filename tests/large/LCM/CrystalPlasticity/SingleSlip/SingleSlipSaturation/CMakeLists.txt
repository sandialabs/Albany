
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipSaturation.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipSaturation.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip.g COPYONLY)	       
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipSaturation_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipSaturation_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipSaturation.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipSaturation.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipSaturation.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipSaturation.gold.exo COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 SET(OUTFILE "SingleSlipSaturation.exo")
 SET(REF_FILE "SingleSlipSaturation.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=SingleSlipSaturation -DTEST_ARGS=SingleSlipSaturation.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
