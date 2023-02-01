#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# Copy Exodus test file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e
                 ${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e.4.0
                 ${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e.4.0 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e.4.1
                 ${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e.4.1 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e.4.2
                 ${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e.4.2 COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/2D_Blk_Test.e.4.3
                 ${CMAKE_CURRENT_BINARY_DIR}/2D_Blk_Test.e.4.3 COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.msh
                 ${CMAKE_CURRENT_BINARY_DIR}/cube.msh COPYONLY)

INCLUDE_DIRECTORIES(
  ${Trilinos_INCLUDE_DIRS}  
  ${Trilinos_TPL_INCLUDE_DIRS} 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/disc
  ${CMAKE_SOURCE_DIR}/src
)

# Files in Albany to be built or are needed
SET(SOURCES
          ./UnitTest_BlockedDOFManager.cpp
          ../Albany_UnitTestMain.cpp
)

SET(HEADERS
)

LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

ADD_EXECUTABLE(
  disc_unit_tester
  ${HEADERS} ${SOURCES}
)

set_target_properties(disc_unit_tester PROPERTIES
  PUBLIC_HEADER "${HEADERS}")

TARGET_LINK_LIBRARIES(disc_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

IF(${PYTHON_TEST})
  SET(SOURCES
            ./UnitTest_BlockedJacobian.cpp
            ../Albany_UnitTestMain.cpp
  )

  ADD_EXECUTABLE(
    blockJacobian_unit_tester
    ${HEADERS} ${SOURCES}
  )

  set_target_properties(blockJacobian_unit_tester PROPERTIES
    PUBLIC_HEADER "${HEADERS}")

  TARGET_LINK_LIBRARIES(blockJacobian_unit_tester albanyLib ${ALB_TRILINOS_LIBS} ${Trilinos_EXTRA_LD_FLAGS})

  set(TESTFILES matrices_comparison.py ${PYTHON_TEST_HELPERS_DIR}/__init__.py ${PYTHON_TEST_HELPERS_DIR}/matrix_reader.py)

  file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)
  ADD_TEST(${testNameRoot}_BlockDiscretization_Jacobian "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/matrices_comparison.py")

  set_tests_properties(${testNameRoot}_BlockDiscretization_Jacobian PROPERTIES ENVIRONMENT "PYTHONPATH=$ENV{PYTHONPATH}")
ENDIF()

ADD_TEST(${testNameRoot}_discretization_Serial ${SERIAL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/disc_unit_tester)

# The parallel test does not run properly.
# It is set up to use IOSS to decompose the exodus file ("Use Serial Mesh = 1"), but
# mpirun -np 2 ./disc_unit_tester
# 
# Throws:
#
# p=0: *** Caught standard std::exception of type 'std::logic_error' :
# 
#  /ascldap/users/gahanse/Codes/Albany/src/utility/Albany_ThyraCrsMatrixFactory.cpp:78:
#  
#  Throw number = 1
#  
#  Throw test that evaluated to true: !isOneToOne(m_domain_vs)
#  
#  [ThyraCrsMatrixFactory] Error! When providing an overlapped domain vs, the domain vs must be one-to-one.
#
ADD_TEST(${testNameRoot}_discretization_Parallel ${PARALLEL_CALL} ${CMAKE_CURRENT_BINARY_DIR}/disc_unit_tester)

