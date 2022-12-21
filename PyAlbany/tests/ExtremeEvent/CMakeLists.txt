##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

file(GLOB TESTFILES expected_* *.py *.yaml )

file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)

IF(ALBANY_DEMO_PDES) #these tests need Thermal 2D problem
  add_test(PyAlbany_ExtremeEvent ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/thermal_steady.py")

  set_tests_properties(PyAlbany_ExtremeEvent
      PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

  IF(ALBANY_MPI)
    add_test(PyAlbany_ExtremeEvent_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/thermal_steady.py")

    set_tests_properties(PyAlbany_ExtremeEvent_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
  ENDIF()

  add_test(PyAlbany_ExtremeEventHessian ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/thermal_steady_hessian.py")

  set_tests_properties(PyAlbany_ExtremeEventHessian
      PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

  IF(ALBANY_MPI)
    add_test(PyAlbany_ExtremeEventHessian_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/thermal_steady_hessian.py")

    set_tests_properties(PyAlbany_ExtremeEventHessian_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
  ENDIF()
ENDIF()
