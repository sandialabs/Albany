##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

file(GLOB TESTFILES *.mm *.py *.yaml )

file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)

add_test(PyAlbany_MatrixOperations ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/matrixOperations.py")

set_tests_properties(PyAlbany_MatrixOperations
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_MatrixOperations_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/matrixOperations.py")

set_tests_properties(PyAlbany_MatrixOperations
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
