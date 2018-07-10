
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlipDislocationDensityTraction.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlipDislocationDensityTraction.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlipDislocationDensity_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlipDislocationDensity_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlip.g
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlip.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlipDislocationDensityTraction.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlipDislocationDensityTraction.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/QuadSlipDislocationDensityTraction.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/QuadSlipDislocationDensityTraction.gold.exo COPYONLY)	       

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 SET(OUTFILE "QuadSlipDislocationDensityTraction.exo")
 SET(REF_FILE "QuadSlipDislocationDensityTraction.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=QuadSlipDislocationDensityTraction -DTEST_ARGS=QuadSlipDislocationDensityTraction.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
