##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if (ALBANY_IFPACK2 AND SEACAS_EXODIFF)
# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)

# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Mechanics.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/Mechanics.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHydrogen.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHydrogen.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHydrogenV2.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHydrogenV2.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHelium.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHelium.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHeliumV2.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHeliumV2.yaml COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsHe.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsHe.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsHeV2.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsHeV2.yaml COPYONLY)

# exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Mechanics.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/Mechanics.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHydrogen.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHydrogen.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHydrogenV2.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHydrogenV2.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHelium.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHelium.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHeliumV2.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHeliumV2.exodiff COPYONLY)

# gold files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Mechanics.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/Mechanics.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHydrogen.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHydrogen.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHydrogenV2.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHydrogenV2.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHelium.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHelium.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHeliumV2.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHeliumV2.gold.e COPYONLY)

# python runtest file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT.cmake COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  #test 1: just Mechanics
  SET(OUTFILE "Mechanics.e")
  SET(REF_FILE "Mechanics.gold.e")
  add_test(NAME ${testName}_JustMechanics
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
           -DTEST_NAME=Mechanics -DTEST_ARGS=Mechanics.yaml -DMPIMNP=1
           -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
           -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtestT.cmake)
  #test 2: Mechanics and Hydrogen
  SET(OUTFILE "MechanicsAndHydrogen.e")
  SET(REF_FILE "MechanicsAndHydrogen.gold.e")
  add_test(NAME ${testName}_MechanicsAndHydrogen
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
           -DTEST_NAME=MechanicsAndHydrogen -DTEST_ARGS=MechanicsAndHydrogen.yaml -DMPIMNP=1
           -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
           -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtestT.cmake)
  #test 3: Mechanics and Hydrogen version 2 
  SET(OUTFILE "MechanicsAndHydrogenV2.e")
  SET(REF_FILE "MechanicsAndHydrogenV2.gold.e")
  add_test(NAME ${testName}_MechanicsAndHydrogenV2
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
           -DTEST_NAME=MechanicsAndHydrogenV2 -DTEST_ARGS=MechanicsAndHydrogenV2.yaml -DMPIMNP=1
           -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
           -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtestT.cmake)
  #test 4: Mechanics and Helium
  SET(OUTFILE "MechanicsAndHelium.e")
  SET(REF_FILE "MechanicsAndHelium.gold.e")
  add_test(NAME ${testName}_MechanicsAndHelium
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
           -DTEST_NAME=MechanicsAndHelium -DTEST_ARGS=MechanicsAndHelium.yaml -DMPIMNP=1
           -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
           -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtestT.cmake)
  #test 5: Mechanics and Helium version 2
  SET(OUTFILE "MechanicsAndHeliumV2.e")
  SET(REF_FILE "MechanicsAndHeliumV2.gold.e")
  add_test(NAME ${testName}_MechanicsAndHeliumV2
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbanyT.exe}"
           -DTEST_NAME=MechanicsAndHeliumV2 -DTEST_ARGS=MechanicsAndHeliumV2.yaml -DMPIMNP=1
           -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
           -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtestT.cmake)

ENDIF()
endif(ALBANY_IFPACK2 AND SEACAS_EXODIFF)
