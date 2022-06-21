//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_Comm.hpp"

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

bool
initializeMPI (std::vector<std::string> stdvec_args) {
    int ierr = 0;
    MPI_Initialized(&ierr);
    if (!ierr) {
      int argc = (int)stdvec_args.size();
      char **argv = new char*[argc+1];
      for (int i = 0; i < argc; ++i) {
          argv[i] = (char*)stdvec_args[i].data();
      }
      argv[argc] = nullptr;

      MPI_Init(&argc, &argv);
      return true;
    }

    return false;
}

RCP_Teuchos_Comm_PyAlbany
getDefaultComm () {
    return Teuchos::DefaultComm<int>::getComm();
}

RCP_Teuchos_Comm_PyAlbany
getTeuchosComm (mpi4py_comm comm) {
    return Teuchos::rcp<const Teuchos_Comm_PyAlbany>(new Teuchos::MpiComm< int >
      (Teuchos::opaqueWrapper(comm.value)));
}

void finalize() {
    MPI_Finalize();
}

PyObject * reduceAll(RCP_Teuchos_Comm_PyAlbany comm, Teuchos::EReductionType reductOp, PyObject * sendObj)
{
    return NULL;
}

void pyalbany_comm(py::module &m) {
    py::enum_<Teuchos::EReductionType>(m, "EReductionType")
        .value("REDUCE_SUM", Teuchos::REDUCE_SUM)
        .value("REDUCE_MIN", Teuchos::REDUCE_MIN)
        .value("REDUCE_MAX", Teuchos::REDUCE_MAX)
        .value("REDUCE_AND", Teuchos::REDUCE_AND)
        .value("REDUCE_BOR", Teuchos::REDUCE_BOR)
        .export_values();

    py::class_<RCP_Teuchos_Comm_PyAlbany>(m, "PyComm")
        .def(py::init<>())
        .def("getRank", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getRank();
        })
        .def("getSize", [](RCP_Teuchos_Comm_PyAlbany &m) {
            return m->getSize();
        })
        .def("reduceAll", [](RCP_Teuchos_Comm_PyAlbany &m, Teuchos::EReductionType reductOp, PyObject * sendObj) {
            return reduceAll(m, reductOp, sendObj);
        });

    m.def("initializeMPI", &initializeMPI, "A function which initializes MPI if not yet iniatialized");
    m.def("getDefaultComm", &getDefaultComm, "A function which returns the default Teuchos communicator");
    m.def("getTeuchosComm", &getTeuchosComm, "A function which returns a Teuchos communicator corresponding to a Mpi4Py communicator");
    m.def("finalize", &finalize, "A function which finalizes MPI (called once at the end if initializeMPI is called)");
}
