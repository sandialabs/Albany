##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if (ALBANY_IFPACK2)
# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_ALGEBRA} ${CMAKE_CURRENT_BINARY_DIR}/algebra)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)

# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Mechanics.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Mechanics.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHydrogen.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHydrogen.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHydrogenV2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHydrogenV2.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHelium.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHelium.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MechanicsAndHeliumV2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MechanicsAndHeliumV2.xml COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsHe.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsHe.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsHeV2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsHeV2.xml COPYONLY)

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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT.py COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(NAME ${testName} COMMAND "python" "runtestT.py")
  set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF}")
ENDIF()
endif()




