# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusion_cse.g
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusion_cse.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusion.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusionMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusionMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusion.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusion.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusion.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusion.gold.e COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 SET(OUTFILE "surfaceDiffusion.e")
 SET(REF_FILE "surfaceDiffusion.gold.e")
 add_test(NAME ${testName}_surfaceDiffusion
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=surfaceDiffusion -DTEST_ARGS=surfaceDiffusion.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})

endif()
