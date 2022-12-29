//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_Timer.hpp"

namespace py = pybind11;

Teuchos::Time createRCPTime(const std::string name) {
    return Teuchos::Time(name);
}

void pyalbany_time(pybind11::module &m) {
    py::class_<Teuchos::Time, Teuchos::RCP<Teuchos::Time>>(m, "Time")
        .def(py::init(&createRCPTime))
        .def("totalElapsedTime",[](Teuchos::Time &m, bool readCurrentTime){
            return m.totalElapsedTime(readCurrentTime);
        }, py::arg("readCurrentTime") = false)
        .def("name", &Teuchos::Time::name)
        .def("start",[](Teuchos::Time &m, bool reset){
            return m.start(reset);
        }, py::arg("reset") = false)
        .def("stop", &Teuchos::Time::stop);

    py::class_<Teuchos::StackedTimer, Teuchos::RCP<Teuchos::StackedTimer>>(m, "StackedTimer")
        .def("accumulatedTime", &Teuchos::StackedTimer::accumulatedTime)
        .def("baseTimerAccumulatedTime",[](Teuchos::StackedTimer &m, const std::string name){
            return m.findBaseTimer(name)->accumulatedTime();
        });
}
