# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusion.g
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusion.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusion.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusionMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusionMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusion.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusion.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusion.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusion.gold.e COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 SET(OUTFILE "channelDiffusion.e")
 SET(REF_FILE "channelDiffusion.gold.e")
 add_test(NAME MechWithHydrogenFastPath_${testName}
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=channelDiffusion -DTEST_ARGS=channelDiffusion.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})

endif()
