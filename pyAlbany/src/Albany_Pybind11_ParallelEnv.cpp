//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_ParallelEnv.hpp"
#include "Kokkos_Core.hpp"

namespace py = pybind11;

Teuchos::RCP<PyParallelEnv> createPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm, int _num_threads, int _num_devices, int _device_id) {
    return Teuchos::rcp<PyParallelEnv>(new PyAlbany::PyParallelEnv(_comm, _num_threads, _num_devices, _device_id));
}

Teuchos::RCP<PyParallelEnv> createDefaultKokkosPyParallelEnv(RCP_Teuchos_Comm_PyAlbany _comm) {
    return Teuchos::rcp<PyParallelEnv>(new PyAlbany::PyParallelEnv(_comm, -1, -1, -1));
}

void pyalbany_parallelenv(py::module &m) {
    py::class_<PyParallelEnv, Teuchos::RCP<PyParallelEnv>>(m, "PyParallelEnv")
        .def(py::init(&createPyParallelEnv))
        .def(py::init(&createDefaultKokkosPyParallelEnv))
        .def("getNumThreads", &PyParallelEnv::getNumThreads)
        .def("getNumDevices", &PyParallelEnv::getNumDevices)
        .def("getDeviceID", &PyParallelEnv::getDeviceID)
        .def("getComm", &PyParallelEnv::getComm)
        .def("setComm", &PyParallelEnv::setComm);
}
