//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PYALBANY_TIMER_H
#define PYALBANY_TIMER_H

#include "Teuchos_StackedTimer.hpp"

#include "Albany_Pybind11_Include.hpp"

using RCP_StackedTimer = Teuchos::RCP<Teuchos::StackedTimer>;
using RCP_Time = Teuchos::RCP<Teuchos::Time>;

void pyalbany_time(pybind11::module &m);

#endif
