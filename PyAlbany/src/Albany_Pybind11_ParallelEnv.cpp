//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_ParallelEnv.hpp"
#include "Kokkos_Core.hpp"

namespace py = pybind11;

RCP_PyParallelEnv createPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm, int _num_threads, int _num_numa, int _device_id) {
    return Teuchos::rcp<PyAlbany::PyParallelEnv>(new PyAlbany::PyParallelEnv(_comm, _num_threads, _num_numa, _device_id));
}

RCP_PyParallelEnv createDefaultKokkosPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm) {
    return Teuchos::rcp<PyAlbany::PyParallelEnv>(new PyAlbany::PyParallelEnv(_comm, -1, -1, -1));
}

void pyalbany_parallelenv(py::module &m) {
    py::class_<RCP_PyParallelEnv>(m, "PyParallelEnv")
        .def(py::init(&createPyParallelEnv))
        .def(py::init(&createDefaultKokkosPyParallelEnv))
        .def("getNumThreads", [](RCP_PyParallelEnv &m) {
            return m->num_threads;
        })
        .def("getNumNuma", [](RCP_PyParallelEnv &m) {
            return m->num_numa;
        })
        .def("getDeviceID", [](RCP_PyParallelEnv &m) {
            return m->device_id;
        })
        .def("getComm", [](RCP_PyParallelEnv &m) {
            return m->comm;
        })
        .def("setComm", [](RCP_PyParallelEnv &m, RCP_Teuchos_Comm_PyAlbany &comm) {
            m->comm = comm;
        });
}
