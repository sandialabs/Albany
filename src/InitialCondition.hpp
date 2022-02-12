//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_INITIAL_CONDITION_HPP
#define ALBANY_INITIAL_CONDITION_HPP

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraTypes.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany {

void InitialConditions (const Teuchos::RCP<Thyra_Vector>& soln,
                        const AbstractDiscretization& disc,
                        Teuchos::ParameterList& icParams,
                        const bool hasRestartSolution);

} // namespace Albany

#endif // ALBANY_INITIAL_CONDITION_HPP
