
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Material_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Material_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Implicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Material_Explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Material_Explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SingleSlipHard_Explicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/SingleSlipHard_Explicit.gold.exo COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 #test 1 - Explicit
 SET(OUTFILE "SingleSlipHard_Explicit.exo")
 SET(REF_FILE "SingleSlipHard_Explicit.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_Explicit
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=SingleSlipHard_Explicit -DTEST_ARGS=SingleSlipHard_Explicit.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
 #test 2 - Implicit
 SET(OUTFILE "SingleSlipHard_Implicit.exo")
 SET(REF_FILE "SingleSlipHard_Implicit.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_Implicit
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=SingleSlipHard_Implicit -DTEST_ARGS=SingleSlipHard_Implicit.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
