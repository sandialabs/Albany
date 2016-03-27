##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubbles.xml
               ${CMAKE_CURRENT_BINARY_DIR}/HeBubbles.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubblesDecay.xml
               ${CMAKE_CURRENT_BINARY_DIR}/HeBubblesDecay.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hexOneElement.g
               ${CMAKE_CURRENT_BINARY_DIR}/hexOneElement.g COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsScaledPlasticity.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsScaledPlasticity.xml COPYONLY)

# exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubbles.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/HeBubbles.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubblesDecay.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/HeBubblesDecay.exodiff COPYONLY)

# gold files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubbles.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/HeBubbles.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/HeBubblesDecay.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/HeBubblesDecay.gold.e COPYONLY)

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




