
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleHexElement.g
               ${CMAKE_CURRENT_BINARY_DIR}/SingleHexElement.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear3d.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear3d.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CrystalElasticity3d_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CrystalElasticity3d_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear3d.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear3d.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShear3d.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShear3d.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnalyticSolution.py
               ${CMAKE_CURRENT_BINARY_DIR}/AnalyticSolution.py COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShearRotated3d.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShearRotated3d.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CrystalElasticityRotated3d_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/CrystalElasticityRotated3d_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShearRotated3d.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShearRotated3d.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ElasticShearRotated3d.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ElasticShearRotated3d.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnalyticSolution.py
               ${CMAKE_CURRENT_BINARY_DIR}/AnalyticSolution.py COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 #test 1 - Elastic shear 3d
 SET(OUTFILE "ElasticShear3d.exo")
 SET(REF_FILE "ElasticShear3d.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_ElasticShear3d
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=ElasticShear3d -DTEST_ARGS=ElasticShear3d.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
 #test 2 - Elastic Shear rotated 3d
 SET(OUTFILE "ElasticShearRotated3d.exo")
 SET(REF_FILE "ElasticShearRotated3d.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_ElasticShearRotated3d
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=ElasticShearRotated3d -DTEST_ARGS=ElasticShearRotated3d.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
