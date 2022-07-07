//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Pybind11_Timer.hpp"

namespace py = pybind11;

RCP_Time createRCPTime(const std::string name) {
    return Teuchos::rcp<Teuchos::Time>(new Teuchos::Time(name));
}

void pyalbany_time(pybind11::module &m) {
    py::class_<RCP_Time>(m, "Time")
        .def(py::init(&createRCPTime))
        .def("totalElapsedTime",[](RCP_Time &m){
            return m->totalElapsedTime();
        })
        .def("name",[](RCP_Time &m){
            return m->name();
        })
        .def("start",[](RCP_Time &m){
            return m->start();
        })
        .def("stop",[](RCP_Time &m){
            return m->stop();
        });

    py::class_<RCP_StackedTimer>(m, "RCPStackedTimer")
        .def(py::init())
        .def("accumulatedTime",[](RCP_StackedTimer &m, const std::string name){
            return m->accumulatedTime(name);
        })
        .def("baseTimerAccumulatedTime",[](RCP_StackedTimer &m, const std::string name){
            return m->findBaseTimer(name)->accumulatedTime();
        });
}
