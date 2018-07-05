
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationOnMesh.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationOnMesh.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OrientationNotOnMesh.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/OrientationNotOnMesh.gold.exo COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 #test 1 - OrientationOnMesh
 SET(OUTFILE "OrientationOnMesh.exo")
 SET(REF_FILE "OrientationOnMesh.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=OrientationOnMesh -DTEST_ARGS=OrientationOnMesh.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
 #test 2 - OrientationOnMesh 4 procs
 SET(OUTFILE "OrientationOnMesh.exo")
 SET(REF_FILE "OrientationOnMesh.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_np4
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
          -DTEST_NAME=OrientationOnMesh -DTEST_ARGS=OrientationOnMesh.yaml -DMPIMNP=${MPIMNP}
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DSEACAS_EPU=${SEACAS_EPU}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
 #test 3 - OrientationNotOnMesh
 SET(OUTFILE "OrientationNotOnMesh.exo")
 SET(REF_FILE "OrientationNotOnMesh.gold.exo")
 add_test(NAME CrystalPlasticity_OrientationNotOnMesh
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=OrientationNotOnMesh -DTEST_ARGS=OrientationNotOnMesh.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
 #test 4 - OrientationNotOnMesh 4 procs
 SET(OUTFILE "OrientationNotOnMesh.exo")
 SET(REF_FILE "OrientationNotOnMesh.gold.exo")
 add_test(NAME CrystalPlasticity_OrientationNotOnMesh_np4
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
          -DTEST_NAME=OrientationNotOnMesh -DTEST_ARGS=OrientationNotOnMesh.yaml -DMPIMNP=${MPIMNP}
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF} -DSEACAS_EPU=${SEACAS_EPU}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
