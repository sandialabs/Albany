//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_Comm.hpp"
#include <mpi4py/mpi4py.h>

namespace py = pybind11;

// The implementation of the conversion from a Mpi4Py communicator 
// to a Teuchos one is based on:
// https://stackoverflow.com/questions/70423477/pybind11-send-mpi-communicator-from-python-to-cpp

struct mpi4py_comm {
  mpi4py_comm() = default;
  mpi4py_comm(MPI_Comm value) : value(value) {}
  operator MPI_Comm () { return value; }

  MPI_Comm value;
};

namespace pybind11 { namespace detail {
  template <> struct type_caster<mpi4py_comm> {
    public:
      PYBIND11_TYPE_CASTER(mpi4py_comm, _("mpi4py_comm"));

      // Python -> C++
      bool load(handle src, bool) {
        if (import_mpi4py() < 0) {
          throw py::error_already_set();
        }

        PyObject *py_src = src.ptr();

        // Check that we have been passed an mpi4py communicator
        if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
          // Convert to regular MPI communicator
          value.value = *PyMPIComm_Get(py_src);
          return !PyErr_Occurred();
        }

        return false;
      }

      // C++ -> Python
      static handle cast(mpi4py_comm src,
                         return_value_policy /* policy */,
                         handle /* parent */)
      {
        // Create an mpi4py handle
        return PyMPIComm_New(src.value);
      }
  };
}}

RCP_Teuchos_Comm_PyAlbany
getTeuchosComm (mpi4py_comm comm) {
    return Teuchos::rcp<const Teuchos_Comm_PyAlbany>(new Teuchos::MpiComm< int >
      (Teuchos::opaqueWrapper(comm.value)));
}

void pyalbany_comm(py::module &m) {
    py::class_<Teuchos_Comm_PyAlbany, Teuchos::RCP<Teuchos_Comm_PyAlbany>>(m, "PyComm")
        .def("getRank", &Teuchos_Comm_PyAlbany::getRank)
        .def("getSize", &Teuchos_Comm_PyAlbany::getSize)
        .def("barrier", &Teuchos_Comm_PyAlbany::barrier);

    m.def("getTeuchosComm", &getTeuchosComm, "A function which returns a Teuchos communicator corresponding to a Mpi4Py communicator");
}
