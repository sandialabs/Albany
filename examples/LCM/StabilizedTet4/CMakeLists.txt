##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_ALGEBRA} ${CMAKE_CURRENT_BINARY_DIR}/algebra)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/Albany)
if (ALBANY_IFPACK2)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
endif()

# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/galerkin.xml
               ${CMAKE_CURRENT_BINARY_DIR}/galerkin.xml COPYONLY)
if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/galerkinT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/galerkinT.xml COPYONLY)
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stabilized.xml
               ${CMAKE_CURRENT_BINARY_DIR}/stabilized.xml COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stabilized-materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/stabilized-materials.xml COPYONLY)

# mesh file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/bar.g
               ${CMAKE_CURRENT_BINARY_DIR}/bar.g COPYONLY)

# exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/galerkin.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/galerkin.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stabilized.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/stabilized.exodiff COPYONLY)

# gold files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/galerkin.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/galerkin.gold.e COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/stabilized.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/stabilized.gold.e COPYONLY)

# python runtest file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)
if (ALBANY_IFPACK2)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestT.py
               ${CMAKE_CURRENT_BINARY_DIR}/runtestT.py COPYONLY)
endif()

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
# FIXME Test broken with new STK
#  add_test(NAME ${testName} COMMAND "python" "runtest.py")
#  set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF}")
ENDIF()




