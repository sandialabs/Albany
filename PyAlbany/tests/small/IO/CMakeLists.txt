##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

add_test(PyAlbany_IO "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/io_PyAlbany.py")

set_tests_properties(PyAlbany_IO
    PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")

IF(ALBANY_MPI)
  add_test(PyAlbany_IO_MPI "mpirun" "-np" "${ALBANY_PYTHON_N_MPI}" "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/io_PyAlbany.py")

  set_tests_properties(PyAlbany_IO_MPI
      PROPERTIES ENVIRONMENT "${PYALBANY_PYTHONPATH}")
ENDIF()
