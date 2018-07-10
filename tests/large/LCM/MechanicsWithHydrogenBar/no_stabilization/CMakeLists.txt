
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
#execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
#                ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusion.exo
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusion.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionNoStab.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionNoStab.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsScaledPlasticity.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsScaledPlasticity.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionNoStab.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionNoStab.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blockDiffusionNoStab.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/blockDiffusionNoStab.gold.e COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
# Big diffs in parallel
 SET(OUTFILE "blockDiffusionNoStab.e")
 SET(REF_FILE "blockDiffusionNoStab.gold.e")
 add_test(NAME MechanicsWithHydrogenBar_${testName}
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=blockDiffusionNoStab -DTEST_ARGS=blockDiffusionNoStab.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})

endif()
