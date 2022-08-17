//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_Comm.hpp"
#include <mpi4py/mpi4py.h>

namespace py = pybind11;

template <typename T>
void def_Teuchos_functions(T m) {
  m.def("getTeuchosComm", [](pybind11::object py_obj) -> Teuchos::RCP<const Teuchos::Comm<int> > {
    if (import_mpi4py() < 0) {
      throw pybind11::error_already_set();
    }
    auto py_src = py_obj.ptr();
    if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
      return Teuchos::rcp<const Teuchos::MpiComm<int>>(new Teuchos::MpiComm<int>
        (Teuchos::opaqueWrapper(*PyMPIComm_Get(py_src))));
    }
    else
      return Teuchos::null;
  }, "A function which returns a Teuchos communicator corresponding to a Mpi4Py communicator");
}


void pyalbany_comm(py::module &m) {
    py::class_<Teuchos_Comm_PyAlbany, Teuchos::RCP<Teuchos_Comm_PyAlbany>>(m, "PyComm")
        .def("getRank", &Teuchos_Comm_PyAlbany::getRank)
        .def("getSize", &Teuchos_Comm_PyAlbany::getSize)
        .def("barrier", &Teuchos_Comm_PyAlbany::barrier);

    def_Teuchos_functions(m);
}
