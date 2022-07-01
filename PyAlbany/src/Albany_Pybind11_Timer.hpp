//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PYALBANY_TIMER_H
#define PYALBANY_TIMER_H

#include "Teuchos_StackedTimer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using RCP_StackedTimer = Teuchos::RCP<Teuchos::StackedTimer>;
using RCP_Time = Teuchos::RCP<Teuchos::Time>;

/**
 * \brief createRCPTime function
 * 
 * This function is used to create an RCP to a new Teuchos::Time
 * associated to the input name.
 */
RCP_Time createRCPTime(const std::string name);

void pyalbany_time(pybind11::module &m);

#endif
