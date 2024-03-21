//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_RCP.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Albany_Pybind11_Comm.hpp"
#include "Albany_Pybind11_ParallelEnv.hpp"
#include "Albany_Pybind11_ParameterList.hpp"
#include "Albany_Pybind11_Tpetra.hpp"
#include "Albany_Pybind11_Timer.hpp"

#include "Albany_Interface.hpp"

namespace py = pybind11;

#ifndef BINDER_PYBIND11_TYPE_CASTER
    #define BINDER_PYBIND11_TYPE_CASTER
    PYBIND11_DECLARE_HOLDER_TYPE(T, Teuchos::RCP<T>)
    PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
    PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
    PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

PYBIND11_MODULE(AlbanyInterface, m) {
    m.doc() = "PyAlbany module";

    pyalbany_comm(m);
    pyalbany_parallelenv(m);
    pyalbany_parameterlist(m);
    pyalbany_map(m);
    pyalbany_vector(m);
    pyalbany_mvector(m);
    pyalbany_crsmatrix(m);
    pyalbany_time(m);

    py::class_<PyAlbany::PyProblem>(m, "PyProblem")
        .def(py::init<std::string, Teuchos::RCP<PyAlbany::PyParallelEnv>>())
        .def(py::init<Teuchos::RCP<Teuchos::ParameterList>, Teuchos::RCP<PyAlbany::PyParallelEnv>>())
        .def("performSolve", &PyAlbany::PyProblem::performSolve)
        .def("performAnalysis", &PyAlbany::PyProblem::performAnalysis)
        .def("getResponseMap", &PyAlbany::PyProblem::getResponseMap)
        .def("getStateMap", &PyAlbany::PyProblem::getStateMap)
        .def("getParameterMap", &PyAlbany::PyProblem::getParameterMap)
        .def("setDirections", &PyAlbany::PyProblem::setDirections)
        .def("setParameter", &PyAlbany::PyProblem::setParameter)
        .def("getParameter", &PyAlbany::PyProblem::getParameter)
        .def("getResponse", &PyAlbany::PyProblem::getResponse)
        .def("getState", &PyAlbany::PyProblem::getState)
        .def("getSensitivity", &PyAlbany::PyProblem::getSensitivity)
        .def("getReducedHessian", &PyAlbany::PyProblem::getReducedHessian)
        .def("reportTimers", &PyAlbany::PyProblem::reportTimers)
        .def("getCumulativeResponseContribution", &PyAlbany::PyProblem::getCumulativeResponseContribution)
        .def("updateCumulativeResponseContributionWeigth", &PyAlbany::PyProblem::updateCumulativeResponseContributionWeigth)
        .def("updateCumulativeResponseContributionTargetAndExponent", &PyAlbany::PyProblem::updateCumulativeResponseContributionTargetAndExponent)
        .def("getCovarianceMatrix", &PyAlbany::PyProblem::getCovarianceMatrix)
        .def("setCovarianceMatrix", &PyAlbany::PyProblem::setCovarianceMatrix)
        .def("getStackedTimer", &PyAlbany::PyProblem::getStackedTimer);

    m.def("getRankZeroMap", &PyAlbany::getRankZeroMap, "A function which return a map where all the entries are owned by the rank 0");
    m.def("scatterMVector", &PyAlbany::scatterMVector, "A function which scatters a multivector");
    m.def("gatherVector", &PyAlbany::gatherVector, "A function which gathers a vector");
    m.def("gatherMVector", &PyAlbany::gatherMVector, "A function which gathers a multivector");
    m.def("orthogTpMVecs", &PyAlbany::orthogTpMVecs, "A function which orthogonalizes multivectors");
    m.def("finalizeKokkos", &PyAlbany::finalizeKokkos, "A function which finalizes Kokkos if it has been previously initialized");
}
