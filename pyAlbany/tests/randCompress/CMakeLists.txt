##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

file(GLOB TESTFILES *.npy *.txt *.py *.yaml )

file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)

add_test(PyAlbany_RSVDsinglePass ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDsinglePass.py")

set_tests_properties(PyAlbany_RSVDsinglePass
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_RSVDsinglePass_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDsinglePass.py")

set_tests_properties(PyAlbany_RSVDsinglePass_MPI
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_RSVDdoublePassSymmetric ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDdoublePassSymmetric.py")

set_tests_properties(PyAlbany_RSVDdoublePassSymmetric
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_RSVDdoublePassSymmetric_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDdoublePassSymmetric.py")

set_tests_properties(PyAlbany_RSVDdoublePassSymmetric_MPI
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_RSVDdoublePassNonSymmetric ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDdoublePassNonSymmetric.py")

set_tests_properties(PyAlbany_RSVDdoublePassNonSymmetric
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_RSVDdoublePassNonSymmetric_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDdoublePassNonSymmetric.py")

set_tests_properties(PyAlbany_RSVDdoublePassNonSymmetric_MPI
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_HODLR ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatHODLR.py")

set_tests_properties(PyAlbany_HODLR
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_HODLR_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatHODLR.py")

set_tests_properties(PyAlbany_HODLR_MPI
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_adjointSolve ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/adjointSolve.py")

set_tests_properties(PyAlbany_adjointSolve
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

add_test(PyAlbany_adjointSolve_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/adjointSolve.py")

set_tests_properties(PyAlbany_adjointSolve_MPI
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

