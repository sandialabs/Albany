
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Implicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit_Material.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit_Material.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlip_Explicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlip_Explicit.gold.exo COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 #test 1 - Explicit
 SET(OUTFILE "SingleSlip_Explicit.exo")
 SET(REF_FILE "SingleSlip_Explicit.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_Explicit
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=SingleSlip_Explicit -DTEST_ARGS=SingleSlip_Explicit.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
 #test 2 - Implicit
 SET(OUTFILE "SingleSlip_Implicit.exo")
 SET(REF_FILE "SingleSlip_Implicit.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_Implicit
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=SingleSlip_Implicit -DTEST_ARGS=SingleSlip_Implicit.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
