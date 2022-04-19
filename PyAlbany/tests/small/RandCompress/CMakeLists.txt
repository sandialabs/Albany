##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

file(GLOB TESTFILES *.npy *.txt *.py *.yaml )

file(COPY ${TESTFILES} ${PyAlbanyPyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)

add_test(PyAlbany_RSVDsinglePass ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDsinglePass.py")

set_tests_properties(PyAlbany_RSVDsinglePass
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
    add_test(PyAlbany_RSVDsinglePass_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDsinglePass.py")

    set_tests_properties(PyAlbany_RSVDsinglePass_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()

add_test(PyAlbany_RSVDdoublePass ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDdoublePass.py")

set_tests_properties(PyAlbany_RSVDdoublePass
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
    add_test(PyAlbany_RSVDdoublePass_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatRSVDdoublePass.py")

    set_tests_properties(PyAlbany_RSVDdoublePass_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()

add_test(PyAlbany_HODLR ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatHODLR.py")

set_tests_properties(PyAlbany_HODLR
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
    add_test(PyAlbany_HODLR_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeatHODLR.py")

    set_tests_properties(PyAlbany_HODLR_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()

