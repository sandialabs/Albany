
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.cmake} ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard.g
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHardRev.g
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHardRev.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Material_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Material_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Material_Implicit_Active_Sets.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Material_Implicit_Active_Sets.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/MultiSlipPlaneHard_Implicit_Active_Sets.gold.exo COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_hard_implicit.py
               ${CMAKE_CURRENT_BINARY_DIR}/plot_hard_implicit.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/examineCondition.py
               ${CMAKE_CURRENT_BINARY_DIR}/examineCondition.py COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# Add the tests
if(ALBANY_IFPACK2)
 #test 1 - Implicit Active Sets
 SET(OUTFILE "MultiSlipPlaneHard_Implicit_Active_Sets.exo")
 SET(REF_FILE "MultiSlipPlaneHard_Implicit_Active_Sets.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_Implicit_Active_Sets
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=MultiSlipPlaneHard_Implicit_Active_Sets -DTEST_ARGS=MultiSlipPlaneHard_Implicit_Active_Sets.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
 #test 2 - Implicit
 SET(OUTFILE "MultiSlipPlaneHard_Implicit.exo")
 SET(REF_FILE "MultiSlipPlaneHard_Implicit.gold.exo")
 add_test(NAME CrystalPlasticity_${testName}_Implicit
          COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
          -DTEST_NAME=MultiSlipPlaneHard_Implicit -DTEST_ARGS=MultiSlipPlaneHard_Implicit.yaml -DMPIMNP=1
          -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
          -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
          -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${runtest.cmake})
endif()
