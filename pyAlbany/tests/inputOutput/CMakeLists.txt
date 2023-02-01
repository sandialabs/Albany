##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

file(GLOB TESTFILES *.npy *.txt *.py *.yaml )

file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)

add_test(PyAlbany_IO ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/io_PyAlbany.py")

set_tests_properties(PyAlbany_IO
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_IO_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/io_PyAlbany.py")

set_tests_properties(PyAlbany_IO_MPI
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
