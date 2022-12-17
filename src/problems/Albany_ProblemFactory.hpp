//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PROBLEM_FACTORY_HPP
#define ALBANY_PROBLEM_FACTORY_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_AbstractFactory.hpp"

namespace Albany {

// A typedef for the base class for AbstractProblem factory classes
using ProblemFactory = AbstractFactory<AbstractProblem,std::string,
                                       const Teuchos::RCP<const Teuchos_Comm>&,
                                       const Teuchos::RCP<Teuchos::ParameterList>&,
                                       const Teuchos::RCP<ParamLib>&>;
} // namespace Albany

#endif // ALBANY_PROBLEM_FACTORY_HPP
