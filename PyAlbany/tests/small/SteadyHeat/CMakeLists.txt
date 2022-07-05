##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

file(GLOB TESTFILES *.npy *.txt *.py *.yaml )

file(COPY ${TESTFILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/.)

add_test(PyAlbany_SteadyHeat ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeat.py")

set_tests_properties(PyAlbany_SteadyHeat
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
    add_test(PyAlbany_SteadyHeat_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeat.py")

    set_tests_properties(PyAlbany_SteadyHeat_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()

add_test(PyAlbany_Inv_SteadyHeat ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/inv_steadyHeat.py")

set_tests_properties(PyAlbany_Inv_SteadyHeat
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
    add_test(PyAlbany_Inv_SteadyHeat_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/inv_steadyHeat.py")

    set_tests_properties(PyAlbany_Inv_SteadyHeat_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()

add_test(PyAlbany_multiple_SteadyHeat ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/multiple_runs_steadyHeat.py")

set_tests_properties(PyAlbany_multiple_SteadyHeat
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
    add_test(PyAlbany_multiple_SteadyHeat_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/multiple_runs_steadyHeat.py")

    set_tests_properties(PyAlbany_multiple_SteadyHeat_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()

add_test(PyAlbany_SteadyHeat_StateMap ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeat_stateMap.py")

set_tests_properties(PyAlbany_SteadyHeat_StateMap
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
    add_test(PyAlbany_SteadyHeat_StateMap_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeat_stateMap.py")

    set_tests_properties(PyAlbany_SteadyHeat_StateMap_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()

add_test(PyAlbany_SteadyHeat_Orthogonalization ${SERIAL_CALL} "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeat_orthog.py")

set_tests_properties(PyAlbany_SteadyHeat_Orthogonalization
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
    add_test(PyAlbany_SteadyHeat_Orthogonalization_MPI "${MPIEX}" "${MPINPF}" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/steadyHeat_orthog.py")

    set_tests_properties(PyAlbany_SteadyHeat_Orthogonalization_MPI
        PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()
