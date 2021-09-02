##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

file(GLOB TESTFILES expected_* thermal_steady.py thermal_steady.yaml )

file(COPY ${TESTFILES} ${PyAlbanyPyFiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)

add_test(PyAlbany_ExtremeEvent ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/thermal_steady.py")

set_tests_properties(PyAlbany_ExtremeEvent
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
  add_test(PyAlbany_ExtremeEvent_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/thermal_steady.py")

  set_tests_properties(PyAlbany_ExtremeEvent_MPI
      PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()
