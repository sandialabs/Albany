//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef INITIAL_CONDITION_HPP
#define INITIAL_CONDITION_HPP

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

void InitialConditions (const Teuchos::RCP<Thyra_Vector>& soln,
                        const Teuchos::RCP<Albany::AbstractDiscretization>& disc,
                        Teuchos::ParameterList& icParams);

} // namespace Albany

#endif // INITIAL_CONDITION_HPP
